"""Download and prepare GPA regression data, then run analysis."""
from pathlib import Path
import runpy
import sys
from sklearn_ml_toolbox.cli import main

if __name__ == "__main__":
    config = Path(__file__).with_name('workflow.toml')
    if not any(arg.split('=')[0] in {'--resume', '--dry-run', '--help', '-h'} for arg in sys.argv[1:]):
        runpy.run_path(str(config.parent / 'download_grades.py'))['download']()
        runpy.run_path(str(config.parent / 'steps/data_generation/generate_dataset.py'))['generate']()
    raise SystemExit(main(['workflow', '--config', str(config), *sys.argv[1:]]))
