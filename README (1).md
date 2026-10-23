# Fake News Detector

A machine learning system that classifies news articles as **True** or **Fake** based on linguistic patterns, built end-to-end from raw data through a deployed, explainable web application.

**Live demo:** FastAPI backend + Streamlit frontend, running locally (see [Setup](#setup) below).

---

## Highlights

- Compared **6 models** across classical ML and deep learning: Naive Bayes, Logistic Regression, LinearSVC, Random Forest, a Bidirectional LSTM, and a fine-tuned DistilBERT transformer.
- Discovered and fixed **three separate instances of data leakage** that were inflating accuracy to a misleading ~99% — including wire-service source tags ("Reuters"), embedded URLs, and a dropped cleaning step in one pipeline — bringing results down to an honest, defensible ~97%.
- Built an **explainability layer** (LIME) so every prediction shows the specific words that drove it.
- Ran a **cross-dataset generalization study** (ISOT, WELFake, LIAR; classical and deep-learning models, multiple seeds): accuracy that is 90-99% on the training dataset falls to roughly 56-67% on a different news dataset. See [Cross-dataset generalization study](#cross-dataset-generalization-study).
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

## Cross-dataset generalization study

The ~97% above is measured on held-out articles from the **same** dataset (ISOT). To find out what happens on a different dataset, the same pipeline and several deep-learning models were trained on one dataset and tested on others. Full write-up: [`docs/Cross_Dataset_Results_Report.pdf`](docs/Cross_Dataset_Results_Report.pdf).

**Setup**

- Datasets: ISOT, WELFake (reduced to the 23,532 articles that are *not* in ISOT) and LIAR (binary version).
- Shared articles were removed before any train/test split: 38,261 of WELFake's 61,793 articles (61.9%) are copies of ISOT articles. The WELFake copy used also had its labels inverted relative to ISOT, which was detected from the duplicated articles and corrected.
- Models: TF-IDF + LinearSVC / Logistic Regression / Naive Bayes, and DistilBERT, BiLSTM, TextCNN and a small transformer trained from scratch (inspired by, not a replication of, a 2025 reference paper).
- Text cleaning modes: *raw*, *delaked* (URLs and wire-service terms removed) and *strict* (also the dateline and all punctuation removed).

**Main result** (accuracy %, DistilBERT, strict text, mean of 3 seeds):

| Trained on \ Tested on | ISOT | WELFake (not in ISOT) |
|---|---|---|
| ISOT | 98.8 | **63.8** |
| WELFake (not in ISOT) | **67.2** | 96.5 |

Always guessing "true" would score 55.4% on ISOT and 56.9% on WELFake, so the off-diagonal cells are only about 7-12 points above guessing. The classical pipeline gives the same picture (97.5 in-domain, 56.6 and 66.6 across).

**What it shows**

- Every model family scored 90-99% in-domain and far lower across datasets, which suggests the problem lies in what the datasets let models learn and not in the architecture.
- A high-looking transfer number (DistilBERT 85.5% from WELFake to ISOT with punctuation kept) dropped to 67.2% when all punctuation was removed, so part of it appears to come from formatting cues.
- LIAR (one-sentence claims) is near chance for every model and is not a fair test for article-style models.

**Caveats:** duplicates were removed by exact match only; "fake" is defined differently in each dataset; seeds vary only training, not the (fixed, capped) test sets; the raw and delaked deep-learning runs use a single seed. These results do **not** test the reference paper's own model.

**Reproduce**

```bash
pip install -r requirements.txt -r requirements-cross-dataset.txt

# classical models
python src/phase7_cross_dataset.py --isot-fake Fake.csv --isot-true True.csv \
    --welfake WELFake_Dataset.csv --welfake-flip --liar-csv liar_binary.csv

# deep-learning models (GPU recommended)
python src/phase8_cross_dataset_dl.py --isot-fake Fake.csv --isot-true True.csv \
    --welfake WELFake_Dataset.csv --welfake-flip --liar-csv liar_binary.csv --mode strict
```

**Try it on real articles.** `src/practical_test.py` trains three quick models (ISOT-only, WELFake-only and both) and compares their verdicts on any articles you give it, with your own labels if you have them. It can also show the words behind each verdict:

```bash
python src/practical_test.py --isot-fake Fake.csv --isot-true True.csv --welfake WELFake_Dataset.csv \
    --welfake-flip --folder articles/ --explain
```

Name files `true_*.txt` or `fake_*.txt` to record your label, or use a CSV with `text` and `label` columns (see [`examples/practical_set_template.csv`](examples/practical_set_template.csv)). It checks writing *style*, not facts.

The notebooks in [`notebooks/`](notebooks/) run these experiments end to end on Kaggle. Per-run results are in [`results/`](results/) (parsed from the run logs).

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
                        #   phase7_cross_dataset.py, phase8_cross_dataset_dl.py: cross-dataset study
                        #   practical_test.py: compare models on real articles
notebooks/              # Kaggle notebooks for the cross-dataset experiments and practical testing
examples/               # CSV template for practical testing
results/                # Per-run results of the cross-dataset experiments (CSV)
docs/                   # Cross-dataset results report (PDF)
models/final/           # Deployed model artifacts (TF-IDF vectorizer + LinearSVC)
ui_shots/               # Demo screenshots
requirements.txt
requirements-cross-dataset.txt   # extras for the deep-learning experiments
README.md
```

---

## Limitations & future work

- Trained on a single historical dataset (2016-2017 US political news) — performance on other time periods, topics, or languages is untested and likely weaker. The cross-dataset study above measured the drop between ISOT and WELFake (roughly 30 points); other languages and time periods remain untested.
- Distinguishes writing *style*, not verified *factual accuracy* — not a substitute for fact-checking.
- LSTM and DistilBERT were trained on reduced sample sizes due to local CPU/RAM constraints; results are directionally meaningful but not exhaustively tuned.
- Future work: larger-scale transformer fine-tuning, multi-domain training data, adversarial robustness testing, and multi-class (not just binary) veracity labels.

---

## Tech stack

Python · scikit-learn · PyTorch · Hugging Face Transformers · LIME · FastAPI · Streamlit · pandas · NLTK
