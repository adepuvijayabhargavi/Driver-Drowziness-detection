# Machine learning: ALERT vs DROWSY

The classic CV rules (EAR/MAR thresholds + temporal counting) are the reliable
backbone of the system. This document describes the **optional supervised ML
add-on** that learns a binary "ALERT / DROWSY" verdict from the features the
pipeline already computes per frame.

The ML layer is strictly a *second opinion*:

* it never changes the state machine and never fires the siren by itself;
* it displays `ML: ENABLED  Prediction: ...  Conf: ...%` on the preview and
  logs (only) when it flags drowsiness at high confidence;
* if the model is missing, invalid or feature-mismatched, the app **runs
  without ML** with an explanatory message — the CV rules never depend on it.

---

## 1. The feature schema (one source of truth)

`src/ml/schema.py` defines the **ordered** list of 10 features and is used
identically by collection, training and the real-time classifier:

| # | column | source in the report |
|---|--------|----------------------|
| 0 | `ear_left` | per-eye EAR of the driver's left eye |
| 1 | `ear_right` | per-eye EAR of the driver's right eye |
| 2 | `ear_mean` | average EAR over both eyes |
| 3 | `mar` | mouth aspect ratio |
| 4 | `perclos` | rolling PERCLOS (%) |
| 5 | `eye_closed_duration` | seconds the eyes have stayed closed |
| 6 | `yawn_duration` | seconds the mouth has stayed open |
| 7 | `pitch` | head pitch (deg, + = down) |
| 8 | `yaw` | head yaw (deg, + = driver's left) |
| 9 | `roll` | head roll (deg, + = driver's left) |

No feature is invented or duplicated: everything is computed by real frames.
An unavailable signal becomes an empty cell (parsed as NaN, e.g. head pose
could not be estimated) — it is **never fabricated into a number**.

**Feature-order consistency is enforced.** The saved model embeds
`feature_names` and the classifier refuses to run against any schema that does
not byte-for-byte match `FEATURE_COLUMNS`, so inference can never feed
features in the wrong order.

Two file formats are involved in a dataset:

* `data/features/drowsiness_dataset.csv` (written by the collector, appended
  across sessions): `timestamp, session_id, <10 features>, label`
* the trainer reads the CSVs and drops invalid rows (NaN in any feature, empty
  session id).

## 2. Collecting a dataset

```bash
pip install -r requirements-ml.txt

# Sit in a normal driving posture and let the camera watch you:
python scripts/collect_ml_data.py --label alert  --samples 300

# Then act drowsy (slow prolonged closures, droopy posture, ...):
python scripts/collect_ml_data.py --label drowsy --samples 300
```

Key options of `scripts/collect_ml_data.py`:

| option | default | meaning |
|--------|---------|---------|
| `--label` | required | `alert` or `drowsy`; the label of the **whole** session |
| `--out` | `data/features/drowsiness_dataset.csv` | dataset path (appended across runs) |
| `--source` | auto-scan | camera index (`0`, `1`, ...) or a video file path. With no `--source` the collector probes indexes `0..N` and uses the FIRST camera that returns a real frame — it never trusts `isOpened()` alone, because a DirectShow device can "open" and then only stream black frames. Startup shows `Testing camera 0... / Camera 0: working / Using camera source: 0`; if none work it prints `No working camera found.` with the indexes tested. |
| `--samples` | 500 | target samples for this session (0 = until stopped) |
| `--stride` | 5 | save every N-th frame — near-identical consecutive frames of one state add no information |
| `--session-id` | auto-unique | override the session grouping key (default: `YYYYMMDD_HHMMSS_xxxxxx`, unique per run) |
| `--detector` | `auto` | `auto`, `face_mesh`, `tasks`, `synthetic` (headless smoke tests — never for real data) |
| `--no-window` | off | run without the live preview |

### Session protocol (important)

A robust dataset is **many separate sessions, not one long recording**. Each
collection run gets a fresh unique `session_id`, which lets the trainer hold
out whole sessions during evaluation. Recommended schedule — alternate the
label so both classes exist in several independent sessions:

```
Alert session 1   ->  python scripts/collect_ml_data.py --label alert  --samples 300
Drowsy session 1  ->  python scripts/collect_ml_data.py --label drowsy --samples 300
Alert session 2   ->  python scripts/collect_ml_data.py --label alert  --samples 300
Drowsy session 2  ->  python scripts/collect_ml_data.py --label drowsy --samples 300
Alert session 3   ->  ...and so on
Drowsy session 3  ->  ...
```

Each run **appends** to the same `drowsiness_dataset.csv`, so you can keep
adding sessions over time. Practical minimums for a *preliminary* model: at
least 3 sessions per class (~1000+ rows), spread over different times of day,
lighting, seating and (with consent) different participants. Anything less is
explicitly labelled preliminary in the results.

### Inspect before you train

`scripts/dataset_summary.py` prints a read-only summary of the real data —
rows, sessions, per-column missing/invalid values, class balance and whether a
session-based split is feasible:

```bash
python scripts/dataset_summary.py            # the collector's default file
python scripts/dataset_summary.py data/features/*.csv
```

> **Demo data is not real data.** `data/features/demo_synthetic_dataset.csv`
> and `data/models/drowsiness_synthetic_demo.joblib` exist **only** to exercise
> the pipeline without a webcam. They are synthetic and their accuracy is
> **not** the project's ML accuracy. Do not evaluate the real model on them.

## 3. Training

```bash
python ml/train.py                          # default dataset + model path
python ml/train.py --data data/features/*.csv --out data/models/my.joblib
```

`ml/train.py`:

1. loads and validates the dataset (reports class distribution, sessions,
   diagnostics; warns when the dataset is too small to be trustworthy);
2. drops invalid rows and **prints exactly how many rows were removed and why**
   — per-column missing/invalid value counts and empty session ids — so the
   dataset is never silently modified;
3. splits the **whole sessions** into train/test with
   `GroupShuffleSplit` — a session that appears in the test set is *never* in
   the training set, so evaluation cannot "remember" a driver it already saw
   (**no data leakage**). It prints exactly which `session_id`s landed in
   TRAIN and TEST and the class distribution of each split, using a fixed
   `--seed` (default 42) so the split is reproducible. If sklearn is missing
   the script falls back to a plain random split with a loud leakage warning;
4. trains three pipelines, each `StandardScaler + model`, with
   `class_weight="balanced"` on all three to handle the ALERT/DROWSY class
   imbalance **without duplicating rows**:
   * **Logistic Regression**,
   * **Random Forest**,
   * **SVM** (via `CalibratedClassifierCV` to keep calibrated probabilities
      now that `SVC(probability=True)` is deprecated);
5. evaluates every model on the held-out **unseen sessions** and prints an
   honest per-model comparison (accuracy, per-class P/R/F1, macro and
   weighted averages), saving the table to `results/model_comparison.csv`:

```
     model           acc  alert P  alert R alert F1 drowsy P drowsy R drowsy F1 macro P macro R macro F1 wted F1
----------------------------------------------------------------------------------------------------------------
logistic_regression  0.556    ...
random_forest        0.925    ...
svm                  0.644    ...
```

   Per-class numbers are shown separately because **DROWSY recall matters**: a
   model that never misses a drowsy driver is more valuable than one with slightly
   higher overall accuracy but a sleepy-driver blind spot. The best model is
   selected by **highest DROWSY F1** (ties broken by macro F1) on the unseen
   sessions;

6. writes plots under `results/`:
   * `confusion_matrix_<model>.png` for every model (labelled `Predicted` /
     `Actual`, ALERT / DROWSY),
   * `random_forest_feature_importance.png` (Random Forest is always trained),
   * and prints the **ranked** feature importances in the terminal;
7. saves the best model as a joblib **bundle**. Use
   `--out data/models/drowsiness_real_{model}.joblib` (the `{model}` placeholder
   is replaced with the chosen model's name) to keep the real model separate
   from the synthetic demo. Default output is `data/models/drowsiness_ml.
   joblib`. If the destination exists, the script asks before overwriting
   (`--force` to skip the prompt).

Small-data honesty: with very few samples or a single session the script says
so up front and the numbers stay real — the accuracy you read is the actual
held-out-session accuracy, not a cherry-picked figure.

## 4. Running with ML inference

```bash
python app.py --ml                          # default data/models/drowsiness_ml.joblib
python app.py --ml --ml-model data/models/drowsiness_real_random_forest.joblib
```

The pipeline builds the schema vector from every frame report and scores it.
If the score says `DROWSY` with confidence ≥ `ml.confidence_threshold`
(default 0.85) the app logs a warning; the preview shows the verdict and
confidence:

```
ML: ENABLED  Prediction: DROWSY  Conf: 92.4%
```

Missing / legacy / mismatched models:

| case | behaviour |
|------|-----------|
| file not found | `ML: ENABLED (no model loaded - CV rules active)` |
| bare estimator without schema (old `scripts/train_classifier.py`) | refuses to run, explains it needs a retrain |
| feature schema mismatch | refuses to run (prevents silent wrong predictions) |

## 5. Model bundle contents

`data/models/drowsiness_real_random_forest.joblib` (or the default
`data/models/drowsiness_ml.joblib`) is a dict with:

```python
{
  "schema_version": 1,
  "model": <sklearn Pipeline: StandardScaler + chosen classifier>,
  "model_name": "logistic_regression" | "random_forest" | "svm",
  "feature_names": ["ear_left", ..., "roll"],
  "preprocessing": {"type": "StandardScaler", "fit_on": "train"},
  "target_names": ["alert", "drowsy"],
  "label_map": {"alert": 0, "drowsy": 1},
  "metrics": {...},            # best model, on unseen sessions
  "all_models": {...},         # every model's metrics (comparison)
  "n_samples": ..., "n_train": ..., "n_test": ...,
  "n_sessions_train": ..., "n_sessions_test": ...,
  "train_session_ids": [...],  # session ids held out for training
  "test_session_ids": [...],   # session ids held out for evaluation
  "train_class_counts": {...}, # class distribution used to train
  "test_class_counts": {...},  # class distribution used to evaluate
  "global_class_counts": {...},
  "class_weights": "balanced",
  "sources": [...], "random_state": ..., "trained_at": ...,
  "selection": "..."
}
```

## 6. Tests and code layout

New modules:

* `src/ml/schema.py` — the shared feature schema + vector/row builders,
* `src/ml/dataset.py` — CSV loading, validation, invalid-row dropping,
  session-group train/test split,
* `src/ml/classifier.py` — bundle-aware inference wrapper,
* `scripts/collect_ml_data.py` — dataset collection,
* `ml/train.py` — training / evaluation / export.

The legacy rolling-window extractor (`src/ml/features.py`) and the old
`scripts/record_session.py` / `scripts/train_classifier.py` remain for
compatibility but are superseded by the schema pipeline above.

All ML logic is exercised by `tests/test_ml_schema.py`,
`tests/test_ml_dataset.py`, `tests/test_ml_classifier.py`,
`tests/test_ml_collect.py`, `tests/test_ml_training.py`,
`tests/test_ml_pipeline.py` and `tests/test_dataset_summary.py` — no webcam
required (`--detector synthetic`).

## 7. Current status and results

All numbers below are **measured** on the real collected dataset
(`data/features/drowsiness_dataset.csv`) using whole held-out sessions.
Nothing is synthetic or fabricated.

### Dataset

| | |
|---|---|
| total samples | **1,107** |
| ALERT | 358 (32.3%) across **4 sessions** |
| DROWSY | 749 (67.7%) across **4 sessions** |
| sessions | 8 unique, 0 invalid rows |

### Session-based split (reproducible, `random_state=42`)

Held out **whole sessions** with `GroupShuffleSplit` — no session appears in
both folds (no data leakage).

| split | sessions | rows | class distribution |
|-------|----------|------|--------------------|
| TRAIN | 6 (3 alert, 3 drowsy) | 787 | alert 338 (42.9%), drowsy 449 (57.1%) |
| TEST | 2 (1 alert, 1 drowsy) | 320 | alert 20 (6.2%), drowsy 300 (93.8%) |

```
TRAIN sessions: alert = [20260914_224030_ce0743, 20260914_224128_3cad4e,
                         20260914_224436_7bfbe7]
                 drowsy = [20260914_224221_3f8124, 20260914_225007_ed6634,
                           20260914_225051_cd27c3]
TEST  sessions: alert = [20260914_224109_f0416e]
                 drowsy = [20260914_224757_e0e888]
```

### Results on unseen sessions

```
     model           acc  alert P  alert R alert F1 drowsy P drowsy R drowsy F1   macro F1  wted F1
--------------------------------------------------------------------------------------------------
logistic_regression 0.556  0.119  0.950   0.211   0.994   0.530    0.691     0.451    0.661
random_forest       0.925  0.452  0.950   0.613   0.996   0.923    0.958     0.786    0.937
svm                 0.644  0.149  1.000   0.260   1.000   0.620    0.765     0.513    0.734
--------------------------------------------------------------------------------------------------
```

Full per-model table: `results/model_comparison.csv`.
Confusion matrices: `results/confusion_matrix_logistic_regression.png`,
`results/confusion_matrix_random_forest.png`, `results/confusion_matrix_svm.png`.

### Feature importance (Random Forest)

`results/random_forest_feature_importance.png` + ranked printout:

| rank | feature | importance |
|------|---------|-----------|
| 1 | `perclos` | 0.402 |
| 2 | `pitch` | 0.171 |
| 3 | `yaw` | 0.099 |
| 4 | `roll` | 0.089 |
| 5 | `mar` | 0.084 |
| 6 | `ear_left` | 0.055 |
| 7 | `ear_mean` | 0.043 |
| 8 | `ear_right` | 0.042 |
| 9 | `yawn_duration` | 0.011 |
| 10 | `eye_closed_duration` | 0.003 |

Eye-closure driven PERCLOS dominates, followed by head-pose angles — consistent
with the CV rules that already fire on prolonged closure and nodding.

### Selected model

**Random Forest** → saved as `data/models/drowsiness_real_random_forest.joblib`.

Selection priority is DROWSY recall, then DROWSY F1, then stability. On unseen
sessions Random Forest scored **DROWSY recall 0.923 / DROWSY F1 0.958** vs
Logistic Regression 0.530 / 0.691 and SVM 0.620 / 0.765. Logistic Regression and
SVM are too close to the drowsy/alert boundary on this data (many borderline
drowsy frames mislabelled alert), which a real system cannot afford. Random
Forest also held the best macro F1 (0.786) and weighted F1 (0.937), giving the
most stable behaviour across both classes. It was **chosen by measured results,
not by assumption**.

### Verify inference

```bash
python app.py --ml --ml-model data/models/drowsiness_real_random_forest.joblib
```

Preview shows `ML: ENABLED  Prediction: DROWSY/ALERT  Conf: XX%`; the confidence
is the genuine probability of the predicted class (the RF bundle exposes
`predict_proba`). If the model is missing or mismatched, the app runs on the CV
rules alone with an explanatory message.

### Limitations (read this first)

* **The test set is just 2 sessions** (1 alert session of 20 rows + 1 drowsy
  session of 300 rows). The alert metrics are therefore high-variance — the
  alert test recall (0.95–1.00) comes from a single short session.
* The test fold is strongly imbalanced (6% alert / 94% drowsy); DROWSY numbers
  are more reliable than ALERT numbers on this held-out set.
* Early/preliminary sessions are short (18–97 samples vs 300) and one participant /
  one capture day / one lighting environment — **participant and environment
  generalization are unverified**.
* `class_weight="balanced"` compensates the 32/68 imbalance inside each fold
  without duplicating rows, but it does not fix an insufficient test set.

> **This is a PRELIMINARY dataset. Larger multi-session / multi-participant
> data is required before the accuracy figures above can be trusted for
> real-world use. Do not claim production-level accuracy.**

### What to do next

1. Collect more sessions (target ≥ 3–5 more per class), ideally with a second
   participant in a different room/lighting, so a future hold-out covers
   unseen drivers.
2. Re-run `python scripts/dataset_summary.py`, then `python ml/train.py
   --force --out "data/models/drowsiness_real_{model}.joblib"`.
3. Re-verify with `python app.py --ml
   --ml-model data/models/drowsiness_real_<best>.joblib`.