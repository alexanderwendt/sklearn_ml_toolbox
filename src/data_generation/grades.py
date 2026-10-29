"""Keep ordered parental scores numeric and encode nominal student categories."""
import pandas as pd


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ('ParentalSupport', 'ParentalEducation'):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
    for column in ("Gender", "Ethnicity", "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering"):
        if column in frame:
            frame[column] = frame[column].astype("str")
    return frame
