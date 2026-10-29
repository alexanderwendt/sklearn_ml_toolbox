"""Run this dataset using the shared CLI and the active Python environment."""
from pathlib import Path
import sys

from sklearn_ml_toolbox.cli import main


if __name__ == '__main__':
    raise SystemExit(main(['workflow', '--config', str(Path(__file__).with_name('workflow.toml')), *sys.argv[1:]]))
