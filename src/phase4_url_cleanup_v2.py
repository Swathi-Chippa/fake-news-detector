from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "delaked_data.csv"
OUTPUT_PATH = PROJECT_DIR / "delaked_data_v2.csv"


URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+\b|\b[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?\b"
)


def get_stop_words() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


STOP_WORDS = get_stop_words()


def strip_urls(text: str) -> str:
    cleaned = str(text).lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\b(?:http|https|www|youtube|youtu|com|org|net|edu|gov|bitly|bit\.ly)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def rebuild_clean_text(raw_text: str) -> str:
    cleaned = strip_urls(raw_text)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = [
        token
        for token in cleaned.split()
        if token not in STOP_WORDS and not re.search(r"(http|www|youtube)", token)
    ]
    return " ".join(tokens)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    expected_cols = {"text", "label", "clean_text"}
    if not expected_cols.issubset(df.columns):
        raise SystemExit(f"Required columns missing. Expected at least {sorted(expected_cols)}")

    original_clean = df["clean_text"].fillna("").astype(str)
    updated_clean = df["text"].fillna("").astype(str).apply(rebuild_clean_text)
    df["clean_text"] = updated_clean

    flagged_mask = original_clean.str.contains("http|www|youtube", case=False, regex=True, na=False)
    retained_flagged = int(flagged_mask.sum())
    residual_flagged = int(df["clean_text"].str.contains("http|www|youtube", case=False, regex=True, na=False).sum())

    df.to_csv(OUTPUT_PATH, index=False)

    print("===== URL Cleanup Applied =====")
    print("Approach: generic URL-stripping on clean_text; no rows removed.")
    print(f"Original rows: {len(df)}")
    print(f"Flagged rows before cleanup: {retained_flagged}")
    print(f"Flagged rows remaining after cleanup: {residual_flagged}")
    print(f"Saved dataset: {OUTPUT_PATH.name}")
    print(f"New shape: {df.shape}")
    print(f"Output size: {OUTPUT_PATH.stat().st_size}")

    if residual_flagged == 0:
        print("Confirmed: no retained rows contain raw http/www/youtube fragments in clean_text.")
    else:
        sample = df.loc[df["clean_text"].str.contains("http|www|youtube", case=False, regex=True, na=False), ["text", "clean_text"]].head(5)
        print("Residual examples:")
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
