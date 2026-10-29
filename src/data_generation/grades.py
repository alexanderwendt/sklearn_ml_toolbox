"""Student categorical codes are labels, not continuous measurements."""
import pandas as pd


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("Gender", "Ethnicity", "ParentalEducation", "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"):
        if column in frame:
            frame[column] = frame[column].astype("str")
    return frame
