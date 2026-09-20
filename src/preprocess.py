"""Dataset loading and text preprocessing."""

import re
import string

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DEFAULT_DATA_SOURCE = (
   "data/fake_or_real_news.csv"
)

LABEL_MAP = {"REAL": 0, "FAKE": 1}
CLASS_NAMES = ["REAL", "FAKE"]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\w*\d+\w*", "", text)
    words = (word for word in text.split() if word not in ENGLISH_STOP_WORDS)
    return " ".join(words)


def load_dataset(source: str = DEFAULT_DATA_SOURCE) -> pd.DataFrame:
    df = pd.read_csv(source)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["text"] = df["text"].fillna("")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    conflicts = df.groupby("text")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Some duplicate articles have conflicting labels.")
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    unknown = set(df["label"].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(
            f"Unexpected labels {sorted(unknown)}; expected {list(LABEL_MAP)}"
        )

    df["class"] = df["label"].map(LABEL_MAP)
    df["clean_text"] = df["text"].apply(clean_text)
    return df[df["clean_text"].str.len() > 0].reset_index(drop=True)
