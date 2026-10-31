"""Download the original student-performance CSV from Kaggle (public dataset)."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

DATASET = 'rabieelkharoua/students-performance-dataset'
URL = f'https://www.kaggle.com/api/v1/datasets/download/{DATASET}'
DEFAULT_DESTINATION = Path(__file__).resolve().parent / 'data_raw/Student_performance_data.csv'


def extract_csv(archive):
    """Select by schema rather than relying on Kaggle's original filename."""
    matches = []
    with ZipFile(BytesIO(archive)) as bundle:
        for name in bundle.namelist():
            if not name.lower().endswith('.csv'):
                continue
            content = bundle.read(name)
            reader = csv.DictReader(StringIO(content.decode('utf-8-sig')))
            if {'StudentID', 'GPA', 'GradeClass', 'ParentalSupport', 'ParentalEducation', 'Ethnicity'} <= set(reader.fieldnames or []):
                if next(reader, None) is None:
                    raise ValueError('Downloaded student CSV is empty')
                matches.append((name, content))
    if len(matches) != 1:
        raise ValueError(f'Expected one student-performance CSV, found {len(matches)}')
    return matches[0]


def download(destination=DEFAULT_DESTINATION, *, overwrite=False):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        print(f'Keeping existing source: {destination} (use --overwrite to replace)')
        return destination
    request = Request(URL, headers={'User-Agent': 'sklearn-ml-toolbox/1.0'})
    with urlopen(request, timeout=120) as response:
        original_name, content = extract_csv(response.read())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(destination)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    metadata = {
        'dataset': DATASET, 'source_url': f'https://www.kaggle.com/datasets/{DATASET}',
        'download_url': URL, 'original_filename': original_name,
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'sha256': sha256(content).hexdigest(),
    }
    destination.with_name('source_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(download(args.destination, overwrite=args.overwrite))


if __name__ == '__main__':
    main()
