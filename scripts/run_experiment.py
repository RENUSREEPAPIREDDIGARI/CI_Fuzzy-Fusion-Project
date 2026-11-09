from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-guided dynamic feature fusion")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None, help="Override dataset root (class-folder images)")
    args = parser.parse_args()
    run_experiment(args.config, args.data)


if __name__ == "__main__":
    main()
