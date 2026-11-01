"""Download sample inputs and prepare isolated run artifacts without training."""
import argparse
from pathlib import Path
import subprocess
import sys


STAGES = {
    'automobile': 'regression.prepare',
    'bank_marketing': 'classification.prepare',
    'credit_default': 'classification.prepare',
    'german_credit': 'classification.prepare',
    'grades': 'regression.prepare',
    'real_estate': 'regression.prepare',
    'debug_omxs30': 'prepare.dimensions',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+', choices=STAGES,
                        help='Sample directories to prepare (each gets a new _reruns folder)')
    parser.add_argument('--dry-run', action='store_true', help='Show plans without downloading or writing data')
    args = parser.parse_args(argv)
    samples = Path(__file__).resolve().parents[1] / 'samples'
    for name in args.datasets:
        command = [sys.executable, str(samples / name / 'run.py'), '--until', STAGES[name]]
        if args.dry_run:
            command.append('--dry-run')
        subprocess.run(command, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
