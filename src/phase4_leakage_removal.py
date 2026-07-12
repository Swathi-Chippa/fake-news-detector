from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "cleaned_data.csv"
OUTPUT_PATH = PROJECT_DIR / "delaked_data.csv"


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


def strip_dateline_prefix(raw_text: str) -> tuple[str, str]:
    text = str(raw_text)
    patterns = [
        r"^[A-Z][A-Z .,'/-]{2,60}\s+\((?:Reuters|AP|AFP|Bloomberg|Xinhua)\)\s*-\s*",
        r"^[A-Z][A-Z .,'/-]{2,60},\s*[A-Z]{2,}\s*-\s*",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return re.sub(pattern, "", text, count=1), match.group(0)
    return text, ""


def remove_leakage_terms(text: str) -> str:
    cleaned = str(text)
    cleaned = cleaned.lower()
    cleaned, _ = strip_dateline_prefix(cleaned.upper())
    cleaned = cleaned.lower()

    phrases = sorted(set(LEAKAGE_PHRASES), key=len, reverse=True)
    for phrase in phrases:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)

    cleaned = re.sub(r"\b(" + "|".join(map(re.escape, DAY_WORDS)) + r")\b", " ", cleaned)
    cleaned = re.sub(r"\b(" + "|".join(map(re.escape, MONTH_WORDS)) + r")\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def rank_features(model_name: str, coef, feature_names, top_n: int = 25) -> None:
    weights = coef[0]
    top_pos_idx = weights.argsort()[-top_n:][::-1]
    top_neg_idx = weights.argsort()[:top_n]

    print(f"\n===== {model_name} feature weights =====")
    print(f"Top {top_n} positive features:")
    for rank, idx in enumerate(top_pos_idx, start=1):
        print(f"{rank:>2}. {feature_names[idx]}\t{weights[idx]:.6f}")

    print(f"Top {top_n} negative features:")
    for rank, idx in enumerate(top_neg_idx, start=1):
        print(f"{rank:>2}. {feature_names[idx]}\t{weights[idx]:.6f}")

    found_terms = [term for term in ["reuters", "said", "getty", "washington", "wire", "ap"] if term in {feature_names[i] for i in top_pos_idx}]
    print("Target leakage terms found in top positive features:")
    print("yes: " + ", ".join(found_terms) if found_terms else "no")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    expected_shape = (38620, 3)
    print(f"Loaded cleaned_data.csv shape: {df.shape}")
    if tuple(df.shape) != expected_shape:
        raise SystemExit(f"Shape mismatch: expected {expected_shape}, found {tuple(df.shape)}")

    if not {"text", "label", "clean_text"}.issubset(df.columns):
        raise SystemExit("Required columns missing from cleaned_data.csv")

    original_clean = df["clean_text"].fillna("").astype(str)
    df["clean_text"] = original_clean.apply(remove_leakage_terms)

    changed_mask = original_clean != df["clean_text"]
    changed_count = int(changed_mask.sum())
    print(f"Rows with changed clean_text after leakage removal: {changed_count}")
    print("Examples of stripped leakage content:")
    examples = df.loc[changed_mask, ["text", "clean_text"]].head(5)
    for idx, row in examples.iterrows():
        before = original_clean.loc[idx]
        after = row["clean_text"]
        print(f"Row {idx}:")
        print(f"  before: {before[:350]}")
        print(f"  after : {after[:350]}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved delaked data to: {OUTPUT_PATH.name}")
    print(f"delaked_data.csv shape: {df.shape}")
    print(f"delaked_data.csv size: {OUTPUT_PATH.stat().st_size}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"].fillna(""))
    y = df["label"]
    print("TF-IDF settings: ngram_range=(1, 2), max_features=5000")
    print(f"TF-IDF matrix shape: {X.shape}")
    print(f"TF-IDF features produced: {len(vectorizer.get_feature_names_out())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Train class balance:")
    print(y_train.value_counts().sort_index().to_string())
    print("Test class balance:")
    print(y_test.value_counts().sort_index().to_string())

    baseline_scores = {
        "LogisticRegression": 0.9858,
        "LinearSVC": 0.9930,
    }

    retrained = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
    }

    for name, model in retrained.items():
        print(f"\n===== {name} (delaked) =====")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Original Phase 3 accuracy: {baseline_scores[name]:.4f}")
        print(f"New accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))
        rank_features(name, model.coef_, vectorizer.get_feature_names_out())

    print("\n===== Side-by-side accuracy comparison =====")
    for name in ["LogisticRegression", "LinearSVC"]:
        print(f"{name}: original {baseline_scores[name]:.4f} -> delaked retrain new accuracy printed above")


if __name__ == "__main__":
    main()
