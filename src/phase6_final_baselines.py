from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "delaked_data_v2.csv"
FINAL_DIR = PROJECT_DIR / "models" / "final"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print("===== Final Baseline Re-run =====")
    print(f"Loaded {INPUT_PATH.name} shape: {df.shape}")
    print(f"Loaded columns: {df.columns.tolist()}")
    if "clean_text" not in df.columns or "label" not in df.columns:
        raise SystemExit("Required columns clean_text and/or label are missing")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"].fillna(""))
    y = df["label"].astype(int)
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

    trained = {}
    metrics_rows = []

    final_linear_svc_artifact = PROJECT_DIR / "models" / "linear_svc_v2_corrected.joblib"
    loaded_linear_svc = None
    if final_linear_svc_artifact.exists():
        try:
            loaded_linear_svc = joblib.load(final_linear_svc_artifact)
            print(f"Loaded existing corrected LinearSVC artifact for comparison: {final_linear_svc_artifact.name}")
        except Exception as exc:
            print(f"Warning: could not load existing LinearSVC artifact for comparison: {exc}")
            loaded_linear_svc = None

    for name, model in models.items():
        print(f"\n===== {name} =====")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4, output_dict=True)
        weighted = report["weighted avg"]

        print(f"Training time: {train_seconds:.2f} seconds")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

        metrics_rows.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision (weighted)": weighted["precision"],
                "Recall (weighted)": weighted["recall"],
                "F1 (weighted)": weighted["f1-score"],
                "Training time (s)": train_seconds,
            }
        )
        trained[name] = model

        if name == "linear_svc" and loaded_linear_svc is not None:
            loaded_pred = loaded_linear_svc.predict(X_test)
            identical = bool((loaded_pred == y_pred).all())
            print(f"Comparison with previously saved corrected LinearSVC artifact: predictions identical = {identical}")

    print("\n===== Consolidated Comparison =====")
    table = pd.DataFrame(metrics_rows)[
        ["Model", "Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
    ]
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = FINAL_DIR / "final_tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)

    artifact_map = {
        "final_naive_bayes.joblib": trained["multinomial_nb"],
        "final_logistic_regression.joblib": trained["logistic_regression"],
        "final_linear_svc.joblib": trained["linear_svc"],
        "final_random_forest.joblib": trained["random_forest"],
    }
    for filename, model in artifact_map.items():
        joblib.dump(model, FINAL_DIR / filename)

    print("\n===== Saved Final Artifacts =====")
    print(f"Saved directory: {FINAL_DIR}")
    print(f"{vectorizer_path.name}\t{vectorizer_path.stat().st_size}")
    for filename in artifact_map:
        path = FINAL_DIR / filename
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
