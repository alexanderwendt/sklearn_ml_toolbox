"""Download the UCI bank source and prepare subscription-classification inputs."""
import argparse
from pathlib import Path
import runpy

FOLDER = Path(__file__).resolve().parent
_shared = runpy.run_path(str(FOLDER.parents[1] / 'scripts/download_samples.py'))


def download(*, overwrite=False):
    return _shared['download_source'](FOLDER.name, FOLDER, overwrite=overwrite)


def generate():
    return _shared['prepare_dataset'](FOLDER.name, FOLDER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    download(overwrite=args.overwrite)
    print(generate())
