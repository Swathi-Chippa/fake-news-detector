from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


API_URL = "http://localhost:8000/predict"


def call_api(text: str) -> dict[str, Any]:
    payload = json.dumps({"text": text}).encode("utf-8")
    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def render_top_terms(top_terms: list[dict[str, Any]]) -> None:
    if not top_terms:
        st.write("No explainability terms returned.")
        return

    for term in top_terms:
        word = term.get("word", "")
        weight = float(term.get("weight", 0.0))
        color = "#198754" if weight >= 0 else "#dc3545"
        st.markdown(
            f"<span style='color:{color}; font-weight:600'>{word}</span>"
            f" <span style='color:#666'>({weight:+.4f})</span>",
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
        .app-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .subtle {
            color: #5f6368;
            margin-top: 0;
        }
        .result-box {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            border: 1px solid rgba(0,0,0,0.08);
            background: linear-gradient(180deg, #ffffff 0%, #f7f8fb 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-title'>Fake News Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtle'>Paste an article body and check it against the local FastAPI model.</p>",
    unsafe_allow_html=True,
)

article_text = st.text_area(
    "Article body",
    height=320,
    placeholder="Paste a news article here...",
)

if st.button("Check Article", type="primary"):
    if not article_text.strip():
        st.error("Please paste article text before checking.")
    else:
        try:
            with st.spinner("Calling FastAPI backend..."):
                result = call_api(article_text)
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0.0)
            top_terms = result.get("top_terms", [])

            color = "#198754" if prediction == "True" else "#dc3545"
            st.markdown(
                f"""
                <div class="result-box">
                    <h3 style="margin-top:0;color:{color};">Veracity: {prediction}</h3>
                    <p><strong>Confidence score:</strong> {confidence:.6f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Top contributing terms")
            render_top_terms(top_terms)
        except HTTPError as exc:
            st.error(f"Backend error: {exc.code} {exc.reason}")
        except URLError as exc:
            st.error(f"Could not reach FastAPI backend at {API_URL}: {exc.reason}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

st.caption(
    "This tool is a research prototype trained on a specific historical news dataset. "
    "Results should not be treated as definitive fact-checking."
)
