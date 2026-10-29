"""Run dataset steps with the installed shared toolbox."""
from pathlib import Path
import sys
from sklearn_ml_toolbox.cli import main

if __name__ == "__main__":
    config = Path(__file__).with_name(sys.argv[1] if len(sys.argv) > 1 else "config.toml")
    raise SystemExit(main(["run", "--config", str(config)]))
