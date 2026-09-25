"""Allow ``python -m src`` from the project root."""

from src.main import run

if __name__ == "__main__":
    raise SystemExit(run())
