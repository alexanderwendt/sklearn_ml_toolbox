"""Download and prepare the five UCI samples without rewriting their configuration."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import os
from io import BytesIO
from urllib.request import urlopen

import pandas as pd

DATASETS = {
    'bank_marketing': {'id': 222, 'task': 'classification', 'drop': ['duration'], 'note': 'Pre-contact subscription prediction: duration excluded. Random split is an IID benchmark, not a future-campaign backtest.'},
    'credit_default': {'id': 350, 'task': 'classification', 'drop': ['ID'], 'note': 'Taiwan credit default benchmark. Demographic predictors require review before real lending use.'},
    'german_credit': {'id': 144, 'task': 'classification', 'drop': [], 'note': 'Historical credit-risk benchmark; macro F1 does not implement the original asymmetric cost matrix.'},
    'real_estate': {'id': 477, 'task': 'regression', 'drop': ['No'], 'note': 'Price per unit area, Taiwan. IID benchmark; nearby properties and dates may correlate.'},
    'automobile': {'id': 10, 'task': 'regression', 'drop': ['symboling', 'normalized-losses'], 'note': 'Vehicle-price regression. Rows missing price excluded. Replaces California Housing for explicit UCI CC BY 4.0 provenance.'},
}


def download_file(url: str, destination: Path, *, overwrite=False, validate=None) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
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
        if validate:
            validate(temporary.read_bytes())
        if overwrite:
            temporary.replace(destination)
        else:
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


def download_source(name: str, folder: Path, *, overwrite=False) -> Path:
    """Cache the original UCI CSV and record its retrieval independently of preparation."""
    definition = DATASETS[name]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = download_file(f'https://archive.ics.uci.edu/api/dataset?id={definition["id"]}', folder / 'source_metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))['data']
    original = folder / 'data_raw/original.csv'
    if original.exists() and not overwrite:
        print(f'Keeping existing source: {original}')
        return original
    target = 'price' if name == 'automobile' else metadata['target_col'][0]

    def validate(content):
        frame = pd.read_csv(BytesIO(content), na_values=['?'])
        if frame.empty or target not in frame:
            raise ValueError(f'Expected nonempty {name} CSV with target {target!r}')

    download_file(metadata['data_url'], original, overwrite=overwrite, validate=validate)
    record = {'download_url': metadata['data_url'], 'repository_url': metadata['repository_url'],
              'retrieved_at': datetime.now(timezone.utc).isoformat(),
              'sha256': sha256(original.read_bytes()).hexdigest()}
    (original.parent / 'download_metadata.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return original


def prepare_dataset(name: str, folder: Path) -> Path:
    """Apply only deterministic schema transformations; fit nothing on the full data."""
    folder = Path(folder)
    definition = DATASETS[name]
    metadata = json.loads((folder / 'source_metadata.json').read_text(encoding='utf-8'))['data']
    original = folder / 'data_raw/original.csv'
    target = 'price' if name == 'automobile' else metadata['target_col'][0]
    frame = pd.read_csv(original, na_values=['?'])
    original_rows = len(frame)
    required = {target, *definition['drop']}
    if not required <= set(frame.columns):
        raise ValueError(f'Missing required {name} columns: {sorted(required - set(frame.columns))}')
    frame = frame.dropna(subset=[target])
    if frame.empty:
        raise ValueError(f'{name} has no labeled rows')
    if definition['task'] == 'regression':
        frame[target] = pd.to_numeric(frame[target], errors='raise')
    else:
        labels = {'bank_marketing': {'yes', 'no'}, 'credit_default': {0, 1}, 'german_credit': {1, 2}}[name]
        if not set(frame[target]) <= labels:
            raise ValueError(f'Unexpected {name} target labels')
    if name == 'automobile':
        counts = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}
        for column in ['num-of-doors', 'num-of-cylinders']:
            values = frame[column].replace(counts)
            if not set(values.dropna()) <= set(counts.values()):
                raise ValueError(f'Unexpected count in {column}')
            frame[column] = pd.to_numeric(values, errors='raise')
    if name == 'bank_marketing':
        frame['previously_contacted'] = (frame['pdays'] != -1).astype(int)
        frame['pdays'] = frame['pdays'].replace(-1, float('nan'))
    # These UCI integer columns are category codes, not continuous quantities.
    categorical_codes = {'X2', 'X3', 'X4', *(f'X{i}' for i in range(6, 12))} if name == 'credit_default' else set()
    # UCI variable metadata distinguishes categorical numeric codes.
    for variable in metadata['variables']:
        if variable['role'] == 'Feature' and (variable['type'] in ('Categorical', 'Binary') or variable['name'] in categorical_codes) and variable['name'] in frame:
            column = variable['name']
            frame[column] = frame[column].map(lambda value: f'category_{value}' if pd.notna(value) else None)
    frame = frame.drop(columns=definition['drop'])
    csv = folder / 'data_raw/data.csv'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=csv.parent, delete=False, mode='w', encoding='utf-8', newline='') as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
        # Repeated preparation is byte-identical; avoid replacing an unchanged
        # CSV that a Windows editor or another reader may currently have open.
        if not csv.exists() or temporary.read_bytes() != csv.read_bytes():
            temporary.replace(csv)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    provenance = {'repository_url': metadata['repository_url'], 'download_url': metadata['data_url'], 'doi': metadata['dataset_doi'], 'creators': metadata['creators'], 'license': 'CC BY 4.0', 'original_sha256': sha256(original.read_bytes()).hexdigest(), 'csv_sha256': sha256(csv.read_bytes()).hexdigest(), 'source_rows': original_rows, 'rows': len(frame), 'removed_unlabeled_rows': original_rows - len(frame), 'target': target, 'task': definition['task'], 'dropped_columns': definition['drop'], 'note': definition['note']}
    (csv.parent / 'preparation_metadata.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'{name}: {len(frame)} rows, {len(frame.columns) - 1} predictors, {definition["task"]} target={target}')
    return csv


def download_dataset(name: str, destination: Path, *, overwrite=False) -> Path:
    folder = Path(destination) / name
    download_source(name, folder, overwrite=overwrite)
    return prepare_dataset(name, folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for name in [args.dataset] if args.dataset else DATASETS:
        print(download_dataset(name, args.destination, overwrite=args.overwrite))


if __name__ == '__main__':
    main()
