"""Machine-learning add-on: feature schema, dataset, and classifiers.

The classic CV rules (EAR/MAR thresholds + temporal counting) are the reliable
backbone of the system. The ML layer is *optional* and built on top of them:
it consumes the per-frame features the pipeline already produces and learns to
separate ALERT from DROWSY.

Modules
-------
schema.py    - the single ordered feature schema shared by collection,
               training and real-time inference.
dataset.py   - CSV loading, validation and session-based (leakage-safe) splits.
classifier.py- bundle-aware inference wrapper for the real-time app.
features.py  - legacy rolling-window feature extractor (kept for compatibility).

Training is done offline via ``scripts/collect_ml_data.py`` (collect data) and
``ml/train.py`` (fit + evaluate + export). See docs/machine_learning.md.
"""
