from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "delaked_data_v2.csv"
MODELS_DIR = PROJECT_DIR / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_v2.joblib"
LINEAR_SVC_PATH = MODELS_DIR / "linear_svc_v2.joblib"
PIPELINE_PATH = MODELS_DIR / "linear_svc_v2_pipeline.joblib"
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
    text = re.sub(r"(?i)\b(?:https?://|www\.)\S+\b|\b[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?\b", " ", text)
    text = re.sub(r"\b(?:http|https|www|youtube|youtu|com|org|net|edu|gov|bitly|bit\.ly)\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        token
        for token in text.split()
        if token not in STOP_WORDS and not re.search(r"(http|www|youtube)", token)
    ]
    return " ".join(tokens)


class RawToCleanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(text) for text in X]


@dataclass
class RawTextModel:
    cleaner: RawToCleanTransformer
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


def train_model(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    X_train = train_df["clean_text"].fillna("").astype(str).tolist()
    X_test = test_df["clean_text"].fillna("").astype(str).tolist()
    y_train = train_df["label"].astype(int).tolist()
    y_test = test_df["label"].astype(int).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_svc = LinearSVC()
    calibrated = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)

    start = time.perf_counter()
    calibrated.fit(X_train_vec, y_train)
    train_seconds = time.perf_counter() - start

    y_pred = calibrated.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("===== Retrain on delaked_data_v2.csv =====")
    print("TF-IDF settings: ngram_range=(1, 2), max_features=5000")
    print(f"Train shape: {X_train_vec.shape}, Test shape: {X_test_vec.shape}")
    print(f"Training time: {train_seconds:.2f} seconds")
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    pipeline = Pipeline(
        [
            ("clean", RawToCleanTransformer()),
            ("tfidf", vectorizer),
            ("clf", calibrated),
        ]
    )

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(calibrated, LINEAR_SVC_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)

    print("Saved artifacts:")
    for path in [VECTORIZER_PATH, LINEAR_SVC_PATH, PIPELINE_PATH]:
        print(f"{path.name}\t{path.stat().st_size}")

    return vectorizer, calibrated, pipeline, test_df.reset_index(drop=True), acc


def pick_examples(test_df: pd.DataFrame, model: RawTextModel):
    texts = test_df["text"].fillna("").astype(str).tolist()
    labels = test_df["label"].astype(int).tolist()
    preds = model.predict(texts)

    correct_true = None
    correct_fake = None
    misclassified = None
    for text, y_true, y_pred in zip(texts, labels, preds):
        if y_true == 1 and y_pred == 1 and correct_true is None:
            correct_true = (text, y_true, y_pred)
        elif y_true == 0 and y_pred == 0 and correct_fake is None:
            correct_fake = (text, y_true, y_pred)
        elif y_true != y_pred and misclassified is None:
            misclassified = (text, y_true, y_pred)
        if correct_true and correct_fake and misclassified:
            break

    if misclassified is None:
        for text, y_true, y_pred in zip(texts, labels, preds):
            if y_true != y_pred:
                misclassified = (text, y_true, y_pred)
                break

    return correct_true, correct_fake, misclassified, texts, labels, preds


def explain_example(explainer, model: RawTextModel, example, title: str, num_features: int = 10):
    if example is None:
        print(f"\n--- {title} ---")
        print("No example available.")
        return

    text, y_true, y_pred = example
    proba = model.predict_proba([text])[0]
    pred = int(np.argmax(proba))
    exp = explainer.explain_instance(text, model.predict_proba, num_features=num_features, labels=[0, 1])
    top = exp.as_list(label=pred)[:num_features]

    print(f"\n--- {title} ---")
    print(f"true_label: {LABEL_NAMES[y_true]} ({y_true})")
    print(f"predicted_label: {LABEL_NAMES[pred]} ({pred})")
    print(f"predicted_probability: {proba[pred]:.4f}")
    print(f"text: {text[:500]}")
    print("LIME top features for predicted class:")
    for feature, weight in top:
        print(f"  {feature}: {weight:+.4f}")
    url_terms = [feature for feature, _ in top if re.search(r"(http|www|youtube)", feature, re.I)]
    print(f"URL-style top terms present: {'yes' if url_terms else 'no'}")
    if url_terms:
        print(f"URL-style terms: {', '.join(url_terms)}")


def main() -> None:
    df = load_data()
    print("===== Phase 6 v2 Retrain =====")
    print(f"Loaded delaked_data_v2.csv shape: {df.shape}")
    print(f"Loaded columns: {df.columns.tolist()}")

    vectorizer, classifier, pipeline, test_df, acc = train_model(df)
    v1_acc = 0.9754
    print("\n===== Comparison =====")
    print(f"v1 delaked test accuracy: {v1_acc:.4f}")
    print(f"v2 test accuracy: {acc:.4f}")
    print(f"Difference: {acc - v1_acc:+.4f}")

    model = RawTextModel(RawToCleanTransformer(), vectorizer, classifier)
    correct_true, correct_fake, misclassified, texts, labels, preds = pick_examples(test_df, model)

    print("\n===== Step 2: LIME Checks =====")
    explainer = LimeTextExplainer(class_names=LABEL_NAMES)
    explain_example(explainer, model, correct_true, "Correct True example")
    explain_example(explainer, model, correct_fake, "Correct Fake example")
    explain_example(explainer, model, misclassified, "Misclassified example")


if __name__ == "__main__":
    main()
