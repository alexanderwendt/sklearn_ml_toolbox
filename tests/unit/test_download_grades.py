from io import BytesIO
from zipfile import ZipFile

import pytest

from scripts.download_grades import extract_csv, download


def archive(files):
    stream = BytesIO()
    with ZipFile(stream, 'w') as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    return stream.getvalue()


def test_download_selects_source_by_schema_without_extracting_paths():
    content = b'StudentID,GPA,GradeClass,ParentalSupport,ParentalEducation,Ethnicity\n1,3,1,2,3,0\n'
    name, result = extract_csv(archive({'../source.csv': content, 'other.csv': b'a,b\n1,2\n'}))
    assert name == '../source.csv'
    assert result == content
    with pytest.raises(ValueError, match='Expected one'):
        extract_csv(archive({'wrong.csv': b'a,b\n1,2\n'}))


def test_download_preserves_existing_source(tmp_path):
    path = tmp_path / 'source.csv'
    path.write_bytes(b'existing')
    assert download(path) == path
    assert path.read_bytes() == b'existing'
