"""Generate regression data and start an isolated workflow."""
from pathlib import Path
import runpy
import sys
from sklearn_ml_toolbox.cli import main

if __name__ == '__main__':
    dataset = Path(__file__).resolve().parent
    if not any(arg.split('=')[0] == '--resume' for arg in sys.argv[1:]):
        runpy.run_path(str(dataset / 'steps/data_generation/generate_dataset.py'))['generate']()
    raise SystemExit(main(['workflow', '--config', str(dataset / 'workflow.toml'), *sys.argv[1:]]))
