from __future__ import annotations

from pathlib import Path
import time
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "cleaned_data.csv"
MODELS_DIR = PROJECT_DIR / "models"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded cleaned_data.csv shape: {df.shape}")
    print(f"Loaded cleaned_data.csv columns: {df.columns.tolist()}")

    expected_shape = (38620, 3)
    if tuple(df.shape) != expected_shape:
        raise SystemExit(f"Shape mismatch: expected {expected_shape}, found {tuple(df.shape)}")

    if "clean_text" not in df.columns or "label" not in df.columns:
        raise SystemExit("Required columns clean_text and/or label are missing")

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

    models = {
        "multinomial_nb": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "linear_svc": LinearSVC(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    trained_models = {}
    for name, model in models.items():
        print(f"\n===== {name} =====")
        start = time.perf_counter()
        with warnings.catch_warnings(record=False):
            model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        y_pred = model.predict(X_test)
        print(f"Training time: {elapsed:.4f} seconds")
        print(classification_report(y_test, y_pred, digits=4))
        trained_models[name] = model

    MODELS_DIR.mkdir(exist_ok=True)
    artifact_paths = {}
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    artifact_paths[vectorizer_path.name] = vectorizer_path.stat().st_size

    for name, model in trained_models.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        artifact_paths[path.name] = path.stat().st_size

    print("\nSaved artifacts:")
    for filename, size in artifact_paths.items():
        print(f"{filename}\t{size}")


if __name__ == "__main__":
    main()
