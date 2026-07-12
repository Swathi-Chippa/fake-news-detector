from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords


PROJECT_DIR = Path(__file__).resolve().parent
FAKE_PATH = PROJECT_DIR / "Fake.csv"
TRUE_PATH = PROJECT_DIR / "True.csv"
OUTPUT_PATH = PROJECT_DIR / "cleaned_data.csv"


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError as exc:
        print(f"UnicodeDecodeError while reading {path.name}: {exc}")
        print("Retrying with cp1252 encoding.")
        return pd.read_csv(path, encoding="cp1252")


def clean_text(text: str, stop_words: set[str]) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in stop_words]
    return " ".join(tokens)


def main() -> None:
    print("Downloading NLTK stopwords corpus...")
    download_ok = nltk.download("stopwords")
    print(f"stopwords download success: {download_ok}")
    stop_words = set(stopwords.words("english"))

    fake_df = load_csv(FAKE_PATH)
    true_df = load_csv(TRUE_PATH)

    fake_df = fake_df.copy()
    true_df = true_df.copy()
    fake_df["label"] = 0
    true_df["label"] = 1

    combined = pd.concat([true_df, fake_df], ignore_index=True)
    print(f"Shape after concat: {combined.shape}")

    combined = combined.drop(columns=["title", "subject", "date"])
    print(f"Shape after dropping title/subject/date: {combined.shape}")

    grouped = combined.groupby("text", dropna=False)["label"].agg(
        total_rows="size",
        unique_labels=lambda values: set(values),
        label_count=lambda values: values.nunique(),
    )
    conflicting = grouped[grouped["label_count"] > 1]
    conflict_text_values = len(conflicting)
    conflict_removed_rows = int((conflicting["total_rows"] - 1).sum())
    same_label_removed_rows = int((grouped[grouped["label_count"] == 1]["total_rows"] - 1).sum())
    print(f"Cross-label conflicting text values: {conflict_text_values}")
    print(f"Rows removed from cross-label conflict groups: {conflict_removed_rows}")
    print(f"Rows removed from same-label duplicate groups: {same_label_removed_rows}")
    print("3 cross-label conflict examples:")
    for idx, (text_value, row) in enumerate(conflicting.head(3).iterrows(), start=1):
        snippet = str(text_value).replace("\n", " ")[:220]
        print(f"Example {idx}:")
        print(f"  text snippet: {snippet}")
        print(f"  labels seen: {sorted(row['unique_labels'])}")
        print(f"  row count in group: {int(row['total_rows'])}")

    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before_dedup - len(combined)
    print(f"Duplicate rows removed based on text: {removed}")
    print(f"Shape after deduplication: {combined.shape}")

    short_rows = combined[combined["text"].fillna("").astype(str).str.len() < 20]
    short_rows_count = len(short_rows)
    short_rows_label_counts = short_rows["label"].value_counts().sort_index()
    print(f"Short text rows (<20 characters): {short_rows_count}")
    print("Short text label breakdown:")
    print(short_rows_label_counts.to_string())
    print("Sample of 5 short text rows (<20 characters):")
    sample = short_rows.head(5).copy()
    sample["text_length"] = sample["text"].fillna("").astype(str).str.len()
    print(sample[["label", "text_length", "text"]].to_string(index=False))

    combined = combined[combined["text"].fillna("").astype(str).str.len() >= 20].reset_index(drop=True)
    print(f"Shape after dropping short text rows: {combined.shape}")

    null_before = int(combined["text"].isna().sum())
    combined["text"] = combined["text"].fillna("")
    null_after = int(combined["text"].isna().sum())
    print(f"fillna('') on text changed missing values from {null_before} to {null_after}")

    combined["clean_text"] = combined["text"].apply(lambda value: clean_text(value, stop_words))

    example_indices = [0, min(1, len(combined) - 1)]
    print("Before/after cleaning examples:")
    for idx in example_indices:
        row = combined.iloc[idx]
        print(f"Row {idx}:")
        print(f"  original: {row['text'][:500]}")
        print(f"  cleaned : {row['clean_text'][:500]}")

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned data to: {OUTPUT_PATH.name}")
    print(f"Cleaned dataframe shape: {combined.shape}")
    print(f"cleaned_data.csv size: {OUTPUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
