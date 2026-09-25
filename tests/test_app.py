"""Smoke tests for the root-level ``app.py`` entry point."""

from __future__ import annotations

import subprocess
import sys


def test_app_module_importable():
    import app  # noqa: F401  (root conftest puts the project root on sys.path)


def test_app_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "app.py", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=".",
    )
    assert result.returncode == 0
    assert "--ear-threshold" in result.stdout
    assert "prog" not in result.stderr  # no crash / no traceback
