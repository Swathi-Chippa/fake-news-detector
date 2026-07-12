from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import nltk
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "delaked_data.csv"
MODELS_DIR = PROJECT_DIR / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_delaked.joblib"
LINEAR_SVC_PATH = MODELS_DIR / "linear_svc_delaked.joblib"
PIPELINE_PATH = MODELS_DIR / "linear_svc_delaked_pipeline.joblib"
LABEL_NAMES = ["Fake", "True"]


def get_stop_words() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


STOP_WORDS = get_stop_words()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOP_WORDS]
    return " ".join(tokens)


class CleanTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(text) for text in X]


@dataclass
class TextModel:
    cleaner: CleanTextTransformer
    vectorizer: TfidfVectorizer
    classifier: CalibratedClassifierCV

    def predict_proba(self, texts):
        cleaned = self.cleaner.transform(texts)
        X = self.vectorizer.transform(cleaned)
        return self.classifier.predict_proba(X)

    def predict(self, texts):
        return np.argmax(self.predict_proba(texts), axis=1)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    expected_cols = {"text", "label", "clean_text"}
    if not expected_cols.issubset(df.columns):
        raise SystemExit(f"Required columns missing. Expected at least {sorted(expected_cols)}")
    return df


def train_or_load_model(df: pd.DataFrame):
    MODELS_DIR.mkdir(exist_ok=True)
    if VECTORIZER_PATH.exists() and LINEAR_SVC_PATH.exists() and PIPELINE_PATH.exists():
        print("Loaded existing delaked LinearSVC artifacts from models/")
        vectorizer = joblib.load(VECTORIZER_PATH)
        classifier = joblib.load(LINEAR_SVC_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        return vectorizer, classifier, pipeline

    print("No delaked LinearSVC artifacts found. Retraining on delaked_data.csv...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    X_train = train_df["clean_text"].fillna("").astype(str).tolist()
    y_train = train_df["label"].astype(int).tolist()
    X_test = test_df["clean_text"].fillna("").astype(str).tolist()
    y_test = test_df["label"].astype(int).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_svc = LinearSVC()
    calibrated = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)

    start = time.perf_counter()
    calibrated.fit(X_train_vec, y_train)
    elapsed = time.perf_counter() - start
    y_pred = calibrated.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("===== Retraining =====")
    print("TF-IDF settings: ngram_range=(1, 2), max_features=5000")
    print(f"Train shape: {X_train_vec.shape}, Test shape: {X_test_vec.shape}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    pipeline = Pipeline(
        [
            ("clean", CleanTextTransformer()),
            ("tfidf", vectorizer),
            ("clf", calibrated),
        ]
    )

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(calibrated, LINEAR_SVC_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)
    print("Saved delaked artifacts:")
    for path in [VECTORIZER_PATH, LINEAR_SVC_PATH, PIPELINE_PATH]:
        print(f"{path.name}\t{path.stat().st_size}")

    return vectorizer, calibrated, pipeline


def print_lime_explanation(explainer, model, text: str, title: str, num_features: int = 10):
    proba = model.predict_proba([text])[0]
    pred = int(np.argmax(proba))
    exp = explainer.explain_instance(text, model.predict_proba, num_features=num_features, labels=[0, 1])
    items = exp.as_list(label=pred)

    print(f"\n--- {title} ---")
    print(f"predicted_label: {LABEL_NAMES[pred]} ({pred})")
    print(f"predicted_probability: {proba[pred]:.4f}")
    print(f"true_probability: {proba[1]:.4f}")
    print("LIME top features for predicted class:")
    for feature, weight in items[:num_features]:
        print(f"  {feature}: {weight:+.4f}")
    return pred, proba, items


def main() -> None:
    df = load_data()
    print("===== Phase 6.1: Explainable AI =====")
    print(f"Loaded delaked_data.csv shape: {df.shape}")
    print(f"Loaded columns: {df.columns.tolist()}")
    print("Choice made: CalibratedClassifierCV wrapping LinearSVC.")
    print("Reason: it preserves the stronger LinearSVC decision boundary while providing predict_proba for LIME.")

    vectorizer, classifier, pipeline = train_or_load_model(df)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    test_df = test_df.reset_index(drop=True)

    test_texts = test_df["text"].fillna("").astype(str).tolist()
    test_labels = test_df["label"].astype(int).tolist()
    test_probs = pipeline.predict_proba(test_texts)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Recreated test split accuracy: {test_acc:.4f}")

    exact_correct_true = None
    exact_correct_fake = None
    misclassified = None
    for text, y_true, y_pred in zip(test_texts, test_labels, test_preds):
        if y_true == 1 and y_pred == 1 and exact_correct_true is None:
            exact_correct_true = (text, y_true, y_pred)
        elif y_true == 0 and y_pred == 0 and exact_correct_fake is None:
            exact_correct_fake = (text, y_true, y_pred)
        elif y_true != y_pred and misclassified is None:
            misclassified = (text, y_true, y_pred)
        if exact_correct_true and exact_correct_fake and misclassified:
            break

    if misclassified is None:
        for text, y_true, y_pred in zip(test_texts, test_labels, test_preds):
            if y_true != y_pred:
                misclassified = (text, y_true, y_pred)
                break

    explainer = LimeTextExplainer(class_names=LABEL_NAMES)

    examples = [
        ("Correct True example", exact_correct_true),
        ("Correct Fake example", exact_correct_fake),
        ("Misclassified example", misclassified),
    ]
    for title, example in examples:
        if example is None:
            print(f"\n--- {title} ---")
            print("No example available.")
            continue
        text, y_true, y_pred = example
        print(f"\nSource true label: {LABEL_NAMES[y_true]} ({y_true})")
        print(f"Source predicted: {LABEL_NAMES[y_pred]} ({y_pred})")
        print(f"Text: {text[:500]}")
        print_lime_explanation(explainer, pipeline, text, title)

    print("\n===== Phase 6.2: Multi-Domain Testing =====")
    hand_written_examples = [
        "Congress reached a late-night deal to fund the government and avoid a shutdown this week.",
        "BREAKING: Secret committee says a hidden ballot scheme proves the election was stolen.",
        "The Commerce Department announced new rules for semiconductor exports to several countries.",
        "Officials say a viral video about a senator resigning was edited and is not authentic.",
        "Scientists and regulators briefed lawmakers on new guidelines for AI-generated political ads.",
    ]

    for idx, text in enumerate(hand_written_examples, start=1):
        proba = pipeline.predict_proba([text])[0]
        pred = int(np.argmax(proba))
        print(f"\nStatement {idx}: {text}")
        print(f"Prediction: {LABEL_NAMES[pred]} ({pred})")
        print(f"Confidence: {proba[pred]:.4f}")
        exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=6, labels=[0, 1])
        print("LIME top features:")
        for feature, weight in exp.as_list(label=pred)[:6]:
            print(f"  {feature}: {weight:+.4f}")

    print("\n===== Honest Readout =====")
    print("The model still appears to key on recurring vocabulary associated with newswire-style phrasing, sensational cues, and specific named-entity patterns.")
    print("If any of the examples are near-perfectly classified, that is consistent with the dataset's strong style/subject correlation, not just leakage.")
    print("The LIME outputs should be treated as a robustness check, not proof that the classifier understands factuality.")


if __name__ == "__main__":
    main()
