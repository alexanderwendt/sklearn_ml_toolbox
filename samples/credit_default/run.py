"""Download, prepare and run the credit-card default workflow."""
from pathlib import Path
import runpy
import sys
from sklearn_ml_toolbox.cli import main

if __name__ == "__main__":
    config = Path(__file__).resolve().with_name('workflow.toml')
    if not any(arg.split('=')[0] in {'--resume', '--help', '-h', '--dry-run'} for arg in sys.argv[1:]):
        preparation = runpy.run_path(str(config.parent / 'download_data.py'))
        preparation['download']()
        preparation['generate']()
    raise SystemExit(main(['workflow', '--config', str(config), *sys.argv[1:]]))
