"""Copy sample files without deleting originals or overwriting conflicts."""
import argparse
from hashlib import file_digest
from pathlib import Path
import shutil
import tempfile
import os


def checksum(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def copy_verified(source: Path, destination: Path) -> dict:
    source, destination = Path(source), Path(destination)
    expected = checksum(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checksum(destination) != expected:
            raise FileExistsError(f"Migration conflict: {destination}; original: {source}")
    else:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            if checksum(temporary) != expected:
                raise OSError(f"Copy checksum mismatch: {source}")
            # Same-directory hardlink publishes a complete file without overwriting.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {'source': str(source.resolve()), 'destination': str(destination.resolve()), 'sha256': expected}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    records = [copy_verified(path, args.destination / path.relative_to(args.source)) for path in args.source.rglob('*') if path.is_file() and '.git' not in path.parts]
    text = json.dumps(records, indent=2)
    manifest = args.destination / 'migration_manifest.json'
    if manifest.exists() and manifest.read_text(encoding='utf-8') != text:
        raise FileExistsError(f'Migration manifest conflict: {manifest}')
    manifest.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
