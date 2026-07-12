from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from lime.lime_text import LimeTextExplainer
from pydantic import BaseModel, Field

from src.phase6_retrain_corrected_v2 import corrected_clean as clean_text


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "final"
VECTORIZER_PATH = MODEL_DIR / "final_tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "final_linear_svc.joblib"
LABEL_NAMES = ["Fake", "True"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Article text to classify")


app = FastAPI(title="Fake News Detector API", version="6.3")


@lru_cache(maxsize=1)
def load_artifacts():
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Missing vectorizer artifact: {VECTORIZER_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


@lru_cache(maxsize=1)
def get_explainer() -> LimeTextExplainer:
    return LimeTextExplainer(class_names=LABEL_NAMES, random_state=42)


def _decision_scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, -50.0, 50.0)
    positive = 1.0 / (1.0 + np.exp(-clipped))
    return np.column_stack([1.0 - positive, positive])


def _make_lime_predict_proba(vectorizer, model):
    def predict_proba(texts):
        cleaned_texts = [clean_text(text) for text in texts]
        features = vectorizer.transform(cleaned_texts)
        scores = np.asarray(model.decision_function(features)).ravel()
        return _decision_scores_to_probabilities(scores)

    return predict_proba


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_article(payload: PredictRequest):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    vectorizer, model = load_artifacts()
    cleaned_text = clean_text(raw_text)
    features = vectorizer.transform([cleaned_text])

    score = float(np.asarray(model.decision_function(features)).ravel()[0])
    prediction_index = 1 if score >= 0 else 0
    prediction_label = LABEL_NAMES[prediction_index]

    explainer = get_explainer()
    predict_proba = _make_lime_predict_proba(vectorizer, model)
    explanation = explainer.explain_instance(
        raw_text,
        predict_proba,
        num_features=8,
        labels=[prediction_index],
        num_samples=1000,
    )
    top_terms = [
        {"word": term, "weight": float(weight)}
        for term, weight in explanation.as_list(label=prediction_index)[:8]
    ]

    return {
        "prediction": prediction_label,
        "confidence": score,
        "top_terms": top_terms,
    }

