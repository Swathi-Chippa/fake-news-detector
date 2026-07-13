# Fake News Detector

A machine learning system that classifies news articles as **True** or **Fake** based on linguistic patterns, built end-to-end from raw data through a deployed, explainable web application.

**Live demo:** FastAPI backend + Streamlit frontend, running locally (see [Setup](#setup) below).

---

## Highlights

- Compared **6 models** across classical ML and deep learning: Naive Bayes, Logistic Regression, LinearSVC, Random Forest, a Bidirectional LSTM, and a fine-tuned DistilBERT transformer.
- Discovered and fixed **three separate instances of data leakage** that were inflating accuracy to a misleading ~99% — including wire-service source tags ("Reuters"), embedded URLs, and a dropped cleaning step in one pipeline — bringing results down to an honest, defensible ~97%.
- Built an **explainability layer** (LIME) so every prediction shows the specific words that drove it.
- Deployed as a working **FastAPI backend + Streamlit frontend**, tested against real held-out articles and hand-written out-of-domain examples to evaluate generalization.

---

## Results

Final, verified accuracy after removing all identified sources of leakage:

| Model | Accuracy | Notes |
|---|---|---|
| Naive Bayes | 91.68% | Simple probabilistic baseline |
| Logistic Regression | 96.86% | Strong linear baseline |
| **LinearSVC** | **97.30%** | **Best traditional ML model — deployed in the live app** |
| Random Forest | 95.47% | Ensemble baseline |
| Bidirectional LSTM | 91.30% | Trained on reduced sample (CPU constraints) |
| DistilBERT | 96.75% | Fine-tuned on GPU (Google Colab) |

### The leakage story

Early models scored a suspiciously perfect ~99% accuracy. Investigating *why* (rather than accepting it) revealed the models had learned shortcuts instead of genuine signal:

1. **Wire-service artifacts**: "Reuters," "Washington," and weekday/timestamp tokens appeared almost exclusively in true articles — because true articles in this dataset are wire-service copy. Stripping these dropped accuracy from ~99% to ~97%, a more honest number.
2. **A dropped cleaning step**: a later data version accidentally reverted this fix, causing accuracy to spike back up before being caught and corrected via LIME inspection.
3. **Embedded URLs**: articles containing bare links (e.g. `youtube.com/watch?v=...`) let models key on "does this contain a link" rather than article content. Detected via LIME, fixed by stripping URLs during preprocessing.

Each fix was verified using **LIME explanations** to confirm the model's top contributing words were genuinely content-related (e.g. "ministry," "spokesman," "statement") rather than artifacts of how the data was scraped.

---

## Real-world generalization test

Beyond the held-out test set, the deployed model was tested on hand-written text **outside the training distribution** to check honesty about its own limits:

- A fabricated wire-style article (fake facts, real Reuters-style formatting) — confidently predicted **True** — revealing the model detects *writing style*, not *factual accuracy*.
- An informal first-person anecdote (true story, casual tone) — predicted **Fake** — because it lacks the formal structure the model associates with "True."
- An out-of-domain tech news snippet — near-zero confidence (~0.11) — the model correctly signaled uncertainty rather than confidently guessing wrong.

**Takeaway:** this is a *style-pattern classifier*, not a fact-checker — an important, explicitly documented limitation rather than an overclaimed capability.

---

## Architecture

```text
Raw CSVs (Kaggle: ISOT Fake/Real News Dataset)
-> Cleaning & deduplication
-> Leakage detection & removal (3 rounds)
-> TF-IDF vectorization (ngram_range=(1,2), 5000 features)
-> Model training & comparison (6 models)
-> LIME explainability layer
-> FastAPI backend (/predict, /health)
-> Streamlit frontend
```

---

## Demo

<img width="1440" height="2600" alt="fake_result" src="https://github.com/user-attachments/assets/48396292-2273-42e2-a64a-a016f4946789" />
<img width="1440" height="2600" alt="true_result" src="https://github.com/user-attachments/assets/ca962c8d-0a54-41d4-aad4-6cc84d8c940d" />


---

## Setup

```bash
# Clone the repo
git clone https://github.com/Swathi-Chippa/fake-news-detector.git
cd fake-news-detector

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Note:** the raw dataset (`Fake.csv`, `True.csv`) is not included in this repo due to size. Download it from Kaggle: [clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), and run the preprocessing scripts in `src/` in order (phase2 → phase6) to regenerate the cleaned/delaked datasets and models.

### Run the app

Terminal 1 — backend:
```bash
python -m uvicorn app.app:app --host 127.0.0.1 --port 8000
```

Terminal 2 — frontend:
```bash
python -m streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Project structure

```text
app/                    # FastAPI backend + Streamlit frontend
src/                    # Data pipeline & model training scripts (phase 2-6)
models/final/           # Deployed model artifacts (TF-IDF vectorizer + LinearSVC)
ui_shots/               # Demo screenshots
requirements.txt
README.md
```

---

## Limitations & future work

- Trained on a single historical dataset (2016-2017 US political news) — performance on other time periods, topics, or languages is untested and likely weaker (see generalization tests above).
- Distinguishes writing *style*, not verified *factual accuracy* — not a substitute for fact-checking.
- LSTM and DistilBERT were trained on reduced sample sizes due to local CPU/RAM constraints; results are directionally meaningful but not exhaustively tuned.
- Future work: larger-scale transformer fine-tuning, multi-domain training data, adversarial robustness testing, and multi-class (not just binary) veracity labels.

---

## Tech stack

Python · scikit-learn · PyTorch · Hugging Face Transformers · LIME · FastAPI · Streamlit · pandas · NLTK
