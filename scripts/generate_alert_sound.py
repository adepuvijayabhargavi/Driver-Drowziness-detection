"""Pre-generate the alert WAV (optional - it is also generated on first run).

Usage::

    python scripts/generate_alert_sound.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts.audio import synthesize_alert_wave
from src.config.settings import DEFAULT_ALERT_SOUND_PATH


def main() -> int:
    path = synthesize_alert_wave(DEFAULT_ALERT_SOUND_PATH, duration=0.8)
    print(f"Wrote siren to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
