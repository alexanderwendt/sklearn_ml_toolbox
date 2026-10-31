"""Compatibility entry point for the grades sample downloader."""
from pathlib import Path
import runpy

_downloader = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'samples/grades/download_grades.py'))
extract_csv = _downloader['extract_csv']
download = _downloader['download']


if __name__ == '__main__':
    _downloader['main']()
