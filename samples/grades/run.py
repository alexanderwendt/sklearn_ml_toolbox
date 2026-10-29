"""Run the complete grades analysis in a new isolated run folder."""
from pathlib import Path
import sys
from sklearn_ml_toolbox.cli import main

if __name__ == "__main__":
    config = Path(__file__).with_name('workflow.toml')
    raise SystemExit(main(['workflow', '--config', str(config), *sys.argv[1:]]))
