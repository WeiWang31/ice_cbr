from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from icecbr.evaluate_exported_predictions import main


if __name__ == "__main__":
    main()
