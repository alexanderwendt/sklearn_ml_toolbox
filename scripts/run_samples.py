"""Run every generated sample config through the installed toolbox."""
import argparse
import json
from pathlib import Path

from sklearn_ml_toolbox.workflow import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    args = parser.parse_args()
    summary = []
    for config in sorted(args.samples.glob('*/config*.toml')):
        run = run_pipeline(config)
        summary.append({'config': str(config), 'run': str(run), 'metrics': json.loads((run/'metrics.json').read_text())})
        print(f'{config.parent.name}: {run}', flush=True)
    (args.samples/'sample_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
