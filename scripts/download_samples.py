"""Download five CC BY 4.0 UCI datasets and generate portable sample steps."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import os
from urllib.request import urlopen

import pandas as pd

DATASETS = {
    'bank_marketing': {'id': 222, 'task': 'classification', 'drop': ['duration'], 'note': 'Pre-contact subscription prediction: duration excluded. Random split is an IID benchmark, not a future-campaign backtest.'},
    'credit_default': {'id': 350, 'task': 'classification', 'drop': ['ID'], 'note': 'Taiwan credit default benchmark. Demographic predictors require review before real lending use.'},
    'german_credit': {'id': 144, 'task': 'classification', 'drop': [], 'note': 'Historical credit-risk benchmark; macro F1 does not implement the original asymmetric cost matrix.'},
    'real_estate': {'id': 477, 'task': 'regression', 'drop': ['No'], 'note': 'Price per unit area, Taiwan. IID benchmark; nearby properties and dates may correlate.'},
    'automobile': {'id': 10, 'task': 'regression', 'drop': ['symboling', 'normalized-losses'], 'note': 'Vehicle-price regression. Rows missing price excluded. Replaces California Housing for explicit UCI CC BY 4.0 provenance.'},
}


def download_file(url: str, destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = None
    try:
        with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            received = 0
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
            expected = response.headers.get('Content-Length')
            if expected is not None and received != int(expected):
                raise OSError(f'Incomplete download: expected {expected} bytes, received {received}')
        os.link(temporary, destination)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return destination


def write_new(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise FileExistsError(f'Generated file conflicts with existing content: {path}')
        return
    path.write_text(text, encoding='utf-8')


def sample_files(folder: Path, target: str, task: str, drop: list[str], hook: str | None = None, filename='config.toml', input_name='data.csv'):
    folder.mkdir(parents=True, exist_ok=True)
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    config = f'[data]\npath = "data_raw/{input_name}"\ntarget = {quote(target)}\ntask = {quote(task)}\ndrop = {quote(drop)}\n'
    if hook:
        config += f'prepare = {quote(hook)}\n'
    config += '\n[split]\nstrategy = "random"\ntest_size = 0.2\n\n[training]\nseed = 42\nmodels = ["linear", "forest"]\nfolds = 3\nsearch_iterations = 3\n\n[output]\ndirectory = "results/' + task + '"\n'
    write_new(folder / filename, config)
    runner = '''"""Run dataset steps with the installed shared toolbox."""
from pathlib import Path
import sys
from sklearn_ml_toolbox.cli import main

if __name__ == "__main__":
    config = Path(__file__).with_name(sys.argv[1] if len(sys.argv) > 1 else "config.toml")
    raise SystemExit(main(["run", "--config", str(config)]))
'''
    write_new(folder / 'run.py', runner)


def download_dataset(name: str, destination: Path) -> Path:
    definition = DATASETS[name]
    folder = Path(destination) / name
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = download_file(f'https://archive.ics.uci.edu/api/dataset?id={definition["id"]}', folder / 'source_metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))['data']
    original = download_file(metadata['data_url'], folder / 'data_raw' / 'original.csv')
    target = 'price' if name == 'automobile' else metadata['target_col'][0]
    frame = pd.read_csv(original, na_values=['?'])
    frame = frame.dropna(subset=[target])
    if definition['task'] == 'regression':
        frame[target] = pd.to_numeric(frame[target], errors='raise')
    # UCI variable metadata distinguishes categorical numeric codes.
    for variable in metadata['variables']:
        if variable['role'] == 'Feature' and variable['type'] in ('Categorical', 'Binary') and variable['name'] in frame:
            column = variable['name']
            frame[column] = frame[column].map(lambda value: f'category_{value}' if pd.notna(value) else None)
    csv = folder / 'data_raw' / 'data.csv'
    write_new(csv, frame.to_csv(index=False))
    sample_files(folder, target, definition['task'], definition['drop'])
    provenance = {'repository_url': metadata['repository_url'], 'download_url': metadata['data_url'], 'doi': metadata['dataset_doi'], 'creators': metadata['creators'], 'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/', 'retrieved': datetime.now(timezone.utc).date().isoformat(), 'original_sha256': sha256(original.read_bytes()).hexdigest(), 'csv_sha256': sha256(csv.read_bytes()).hexdigest(), 'rows': len(frame), 'target': target, 'task': definition['task'], 'note': definition['note']}
    write_new(folder / 'provenance.json', json.dumps(provenance, indent=2) + '\n')
    write_new(folder / 'README.md', f'# {metadata["name"]}\n\nSource: {metadata["repository_url"]}\n\nAttribution: {", ".join(metadata["creators"])}. DOI: {metadata["dataset_doi"]}. License: CC BY 4.0.\n\n{definition["note"]}\n\nRun `python run.py` after installing toolbox. Open generated `results/{definition["task"]}/<run>/report.html`. Original CSV and checksum retained; categorical codes converted using source metadata.\n')
    return csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASETS)
    args = parser.parse_args()
    for name in [args.dataset] if args.dataset else DATASETS:
        print(download_dataset(name, args.destination))


if __name__ == '__main__':
    main()
