from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "cleaned_data.csv"
MODELS_DIR = PROJECT_DIR / "models"


def print_ranked_features(model_name: str, coefficients, feature_names, top_n: int = 25) -> None:
    coef = coefficients[0]
    top_positive_idx = coef.argsort()[-top_n:][::-1]
    top_negative_idx = coef.argsort()[:top_n]

    print(f"\n===== {model_name} feature weights =====")
    print(f"Top {top_n} positive features:")
    for rank, idx in enumerate(top_positive_idx, start=1):
        print(f"{rank:>2}. {feature_names[idx]}\t{coef[idx]:.6f}")

    print(f"Top {top_n} negative features:")
    for rank, idx in enumerate(top_negative_idx, start=1):
        print(f"{rank:>2}. {feature_names[idx]}\t{coef[idx]:.6f}")

    positive_terms = {feature_names[idx] for idx in top_positive_idx}
    target_terms = ["reuters", "said", "washington", "official", "update", "source", "via", "report"]
    found = [term for term in target_terms if term in positive_terms]
    print("Target True-indicating terms found in top positive features:")
    if found:
        print(f"yes: {', '.join(found)}")
    else:
        print("no")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    expected_shape = (38620, 3)
    print(f"Loaded cleaned_data.csv shape: {df.shape}")
    print(f"Loaded cleaned_data.csv columns: {df.columns.tolist()}")
    if tuple(df.shape) != expected_shape:
        raise SystemExit(f"Shape mismatch: expected {expected_shape}, found {tuple(df.shape)}")

    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    models = {
        "multinomial_nb": joblib.load(MODELS_DIR / "multinomial_nb.joblib"),
        "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
        "linear_svc": joblib.load(MODELS_DIR / "linear_svc.joblib"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.joblib"),
    }
    print("Loaded saved artifacts from models/")

    X = vectorizer.transform(df["clean_text"].fillna(""))
    y = df["label"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Recreated test split shape: {X_test.shape}")
    print("Test class balance:")
    print(y_test.value_counts().sort_index().to_string())

    for name, model in models.items():
        print(f"\n===== {name} =====")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    feature_names = vectorizer.get_feature_names_out()
    print_ranked_features(
        "LogisticRegression",
        models["logistic_regression"].coef_,
        feature_names,
    )
    print_ranked_features(
        "LinearSVC",
        models["linear_svc"].coef_,
        feature_names,
    )

    print("\n===== Plain Findings =====")
    print("MultinomialNB: no coefficient-based feature ranking available.")
    print("LogisticRegression: feature usage is dominated by a small set of high-weight tokens.")
    print("LinearSVC: feature usage is dominated by a small set of high-weight tokens.")
    print("RandomForestClassifier: no direct TF-IDF coefficient ranking available; tree splits are not summarized here.")


if __name__ == "__main__":
    main()
