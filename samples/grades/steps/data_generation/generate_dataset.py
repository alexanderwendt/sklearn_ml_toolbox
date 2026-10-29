"""Generate regression inputs without GradeClass or student identifiers."""
from pathlib import Path
import pandas as pd


def generate(source=None, output=None):
    dataset = Path(__file__).resolve().parents[2]
    source = Path(source) if source else dataset / 'data_raw/Student_performance_data.csv'
    output = Path(output) if output else dataset / 'data_raw/grades.csv'
    frame = pd.read_csv(source)
    if not {'GradeClass', 'GPA', 'StudentID'}.issubset(frame.columns):
        raise ValueError('Expected GradeClass, GPA and StudentID in source data')
    frame = frame.drop(columns=['GradeClass', 'StudentID'])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


if __name__ == '__main__':
    print(generate())
