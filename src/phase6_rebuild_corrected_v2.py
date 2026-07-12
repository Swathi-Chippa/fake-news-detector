from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "delaked_data.csv"
CURRENT_V2_PATH = PROJECT_DIR / "delaked_data_v2.csv"
OUTPUT_PATH = PROJECT_DIR / "delaked_data_v2.csv"


LEAKAGE_PHRASES = [
    "washington reuters",
    "told reuters",
    "told reporters",
    "featured image",
    "image via",
    "getty images",
    "twitter com",
    "daily mail",
    "reuters",
    "said",
    "via",
    "read",
    "featured",
    "image",
    "getty",
    "images",
    "com",
    "wire",
    "watch",
    "ap",
    "us",
    "gop",
    "hillary",
    "mr",
    "obama",
    "rep",
    "sen",
]

DAY_WORDS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

MONTH_WORDS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
]

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


def remove_leakage_terms(text: str) -> str:
    cleaned = str(text).lower()

    phrases = sorted(set(LEAKAGE_PHRASES), key=len, reverse=True)
    for phrase in phrases:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)

    cleaned = re.sub(r"\b(" + "|".join(map(re.escape, DAY_WORDS)) + r")\b", " ", cleaned)
    cleaned = re.sub(r"\b(" + "|".join(map(re.escape, MONTH_WORDS)) + r")\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def standard_clean(text: str) -> str:
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = [
        token
        for token in cleaned.split()
        if token not in STOP_WORDS and not re.search(r"(http|www|youtube)", token)
    ]
    return " ".join(tokens)


def count_real_words_after_url_strip(text: str) -> int:
    cleaned = strip_urls(text)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = [token for token in cleaned.split() if token]
    return len(tokens)


def rebuild_clean_text(raw_text: str) -> str:
    return standard_clean(remove_leakage_terms(strip_urls(raw_text)))


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    expected_cols = {"text", "label", "clean_text"}
    if not expected_cols.issubset(df.columns):
        raise SystemExit(f"Required columns missing. Expected at least {sorted(expected_cols)}")

    current_v2 = pd.read_csv(CURRENT_V2_PATH)
    if len(current_v2) != len(df):
        print(f"Warning: current v2 row count {len(current_v2)} differs from source {len(df)}.")

    before_clean = current_v2["clean_text"].fillna("").astype(str)
    old_flagged = int(before_clean.str.contains("http|www|youtube", case=False, regex=True, na=False).sum())

    source = df.copy()
    source["url_word_count"] = source["text"].fillna("").astype(str).apply(count_real_words_after_url_strip)
    removed_mask = source["url_word_count"] < 20
    removed_rows = int(removed_mask.sum())
    source["corrected_clean_text"] = source["text"].fillna("").astype(str).apply(rebuild_clean_text)
    source["corrected_removed"] = removed_mask

    retained = source.loc[~removed_mask].copy().reset_index(drop=True)
    retained["clean_text"] = retained["corrected_clean_text"]

    if len(current_v2) != len(source):
        raise SystemExit("Current v2 file and source data are not aligned row-for-row.")

    changed_rows = int(
        ((source["corrected_clean_text"] != current_v2["clean_text"].fillna("").astype(str)) | source["corrected_removed"]).sum()
    )

    residual_flagged = int(retained["clean_text"].str.contains("http|www|youtube", case=False, regex=True, na=False).sum())

    retained = retained[["text", "label", "clean_text"]]
    retained.to_csv(OUTPUT_PATH, index=False)

    print("===== Corrected v2 Rebuild =====")
    print("Chained cleaning steps:")
    print("1. URL/www/youtube stripping")
    print("2. Phase 4.4 leakage-term removal")
    print("3. Standard lowercase/punctuation/stopword cleaning")
    print(f"Rows removed for <20 words after URL stripping: {removed_rows}")
    print(f"Original rows: {len(df)}")
    print(f"New rows: {len(retained)}")
    print(f"New shape: {retained.shape}")
    print(f"Changed rows vs current v2 file: {changed_rows if changed_rows >= 0 else 'n/a'}")
    print(f"Flagged rows in current v2 before correction: {old_flagged}")
    print(f"Residual flagged rows after correction: {residual_flagged}")
    print(f"Saved dataset: {OUTPUT_PATH.name}")
    print(f"Output size: {OUTPUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
