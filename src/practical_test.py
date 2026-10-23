"""
practical_test.py
=================
Try the fake-news models on REAL articles you choose, and see where they disagree.

It trains three quick models (TF-IDF + LinearSVC, your project's final cleaning pipeline):
    ISOT           trained on ISOT only            (what the deployed app uses)
    WELFake-novel  trained on WELFake articles that are not in ISOT
    Both           trained on the two together
and then predicts every article you give it. If you also give the true label, it reports
how often each model was right, so you can see which one holds up on real news.

Models are trained once and saved to --models-dir (about 5 minutes on CPU), then re-used.

Ways to give it articles (combine freely)
-----------------------------------------
  --folder articles/    every .txt file. Name files  true_*.txt / real_*.txt  or  fake_*.txt
                        and the label is picked up from the file name (optional).
  --csv my_set.csv      columns: text, label (optional; 1/true/real or 0/fake/false), source (optional)
  --text "..."          one article pasted on the command line
  --interactive         paste articles one at a time (finish each with a line containing only END)

Example
-------
  python practical_test.py --isot-fake Fake.csv --isot-true True.csv \\
      --welfake WELFake_Dataset.csv --welfake-flip --folder articles/ --explain

Read the output as a style check, not a fact check: the models only see how an article is
written. A well-written false article can score as True, and an informal true one as Fake.
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import phase7_cross_dataset as p7

CHAR_LIMIT = 5000
MODEL_NAMES = ["ISOT", "WELFake-novel", "Both"]
TRUE_WORDS = {"1", "true", "real", "t", "yes"}
FAKE_WORDS = {"0", "fake", "false", "f", "no"}


def prep(text: str) -> str:
    return p7.delaked_clean(str(text)[:CHAR_LIMIT])


def parse_label(value) -> float:
    s = str(value).strip().lower()
    if s in TRUE_WORDS:
        return 1.0
    if s in FAKE_WORDS:
        return 0.0
    return float("nan")


# --------------------------------------------------------------------------
# Training / loading
# --------------------------------------------------------------------------
def train_models(args) -> dict:
    if not (args.isot_fake and args.isot_true and args.welfake):
        raise SystemExit("Models not found in --models-dir. Give --isot-fake, --isot-true and --welfake "
                         "once so they can be trained (they are saved afterwards).")
    raw = {
        "ISOT": p7.finalize(p7.load_isot(args.isot_fake, args.isot_true), "isot"),
        "WELFake": p7.finalize(p7.load_welfake(args.welfake, args.welfake_flip), "welfake"),
    }
    print("\n=== Removing articles shared between ISOT and WELFake ===")
    p7.dedup_across(raw)
    data = {"ISOT": raw["ISOT"], "WELFake-novel": raw["WELFake"]}
    for name, df in data.items():
        print(f"cleaning {name} ({len(df)} articles) ...", flush=True)
        df["clean"] = df["text"].map(prep)
    data["Both"] = pd.concat([data["ISOT"], data["WELFake-novel"]], ignore_index=True)

    models = {}
    for name in MODEL_NAMES:
        df = data[name]
        df = df[df["clean"].str.len() > 0]
        print(f"training {name} on {len(df)} articles ...", flush=True)
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=5000), LinearSVC())
        pipe.fit(df["clean"], df["label"])
        models[name] = pipe
    args.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, args.models_dir / "practical_models.joblib")
    print(f"saved models to {args.models_dir / 'practical_models.joblib'}")
    return models


def get_models(args) -> dict:
    path = args.models_dir / "practical_models.joblib"
    if path.exists() and not args.retrain:
        print(f"loading saved models from {path}")
        return joblib.load(path)
    return train_models(args)


# --------------------------------------------------------------------------
# Collecting articles
# --------------------------------------------------------------------------
def label_from_name(name: str) -> float:
    n = name.lower()
    if n.startswith(("true", "real")):
        return 1.0
    if n.startswith("fake"):
        return 0.0
    return float("nan")


def collect(args) -> pd.DataFrame:
    rows = []
    if args.folder:
        for f in sorted(Path(args.folder).glob("*.txt")):
            rows.append(dict(source=f.name, text=f.read_text(encoding="utf-8", errors="replace"),
                             label=label_from_name(f.name)))
    if args.csv:
        df = pd.read_csv(args.csv)
        if "text" not in df.columns:
            raise SystemExit(f"{args.csv} needs a 'text' column, found {list(df.columns)}")
        for i, r in df.iterrows():
            lab = parse_label(r["label"]) if "label" in df.columns and pd.notna(r["label"]) else float("nan")
            src = str(r["source"]) if "source" in df.columns and pd.notna(r["source"]) else f"csv row {i + 1}"
            rows.append(dict(source=src, text=str(r["text"]), label=lab))
    if args.text:
        rows.append(dict(source="--text", text=args.text, label=float("nan")))
    return pd.DataFrame(rows, columns=["source", "text", "label"])


def read_interactive() -> str | None:
    print("\nPaste an article, then type END on its own line (or type quit):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not lines and line.strip().lower() == "quit":
            return None
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines) if lines else None


# --------------------------------------------------------------------------
# Predicting / explaining
# --------------------------------------------------------------------------
def score_articles(models: dict, texts: list[str]) -> pd.DataFrame:
    cleaned = [prep(t) for t in texts]
    out = {}
    for name, pipe in models.items():
        out[name] = pipe.decision_function(cleaned)       # > 0 means "True", < 0 means "Fake"
    return pd.DataFrame(out)


def explain(models: dict, model_name: str, text: str, k: int = 6) -> str:
    pipe = models[model_name]
    vec, clf = pipe.steps[0][1], pipe.steps[1][1]
    row = vec.transform([prep(text)])
    contrib = row.multiply(clf.coef_[0]).toarray()[0]
    names = np.array(vec.get_feature_names_out())
    top_true = [f"{names[i]} ({contrib[i]:+.2f})" for i in np.argsort(contrib)[::-1][:k] if contrib[i] > 0]
    top_fake = [f"{names[i]} ({contrib[i]:+.2f})" for i in np.argsort(contrib)[:k] if contrib[i] < 0]
    return (f"      words pushing toward TRUE: {', '.join(top_true) or '-'}\n"
            f"      words pushing toward FAKE: {', '.join(top_fake) or '-'}")


def verdict(score: float) -> str:
    return f"{'True' if score > 0 else 'Fake'} ({score:+.2f})"


def report(models: dict, df: pd.DataFrame, args) -> None:
    if df.empty:
        print("No articles to test.")
        return
    scores = score_articles(models, df["text"].tolist())
    words = df["text"].str.split().str.len()
    names = list(models)

    print("\n" + "=" * 100)
    print("RESULTS   (score > 0 means the model says True, < 0 means Fake; the number is a margin, not a probability)")
    print("=" * 100)
    for i, r in df.reset_index(drop=True).iterrows():
        preds = {n: scores.loc[i, n] > 0 for n in names}
        agree = len(set(preds.values())) == 1
        truth = "" if np.isnan(r["label"]) else f"   [your label: {'True' if r['label'] == 1 else 'Fake'}]"
        snippet = textwrap.shorten(" ".join(str(r["text"]).split()), width=90, placeholder="...")
        print(f"\n#{i + 1}  {r['source']}{truth}")
        print(f"      \"{snippet}\"")
        if words.iloc[i] < 50:
            print(f"      note: only {words.iloc[i]} words. The models were trained on full articles, "
                  f"so short texts are unreliable.")
        for n in names:
            mark = ""
            if not np.isnan(r["label"]):
                mark = "   OK" if preds[n] == (r["label"] == 1) else "   WRONG"
            print(f"      {n:<14} {verdict(scores.loc[i, n]):<16}{mark}")
        if not agree:
            print("      -> the models DISAGREE on this article")
        if args.explain:
            print(explain(models, args.explain_model, r["text"]))

    labelled = df["label"].notna().to_numpy()
    if labelled.any():
        y = df.loc[labelled, "label"].to_numpy().astype(int)
        print("\n" + "=" * 100)
        print(f"SUMMARY on {int(labelled.sum())} articles with your labels "
              f"({int(y.sum())} true, {int((1 - y).sum())} fake)")
        print("=" * 100)
        rows = []
        for n in names:
            pred = (scores.loc[labelled, n].to_numpy() > 0).astype(int)
            rows.append(dict(model=n, correct=f"{int((pred == y).sum())}/{len(y)}",
                             accuracy=f"{(pred == y).mean() * 100:.0f}%",
                             balanced_acc=(f"{balanced_accuracy_score(y, pred) * 100:.0f}%"
                                           if len(set(y)) == 2 else "n/a (one class only)"),
                             says_True=f"{pred.mean() * 100:.0f}%"))
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nsays_True = share of your articles the model called True. With only a handful of articles "
              "these numbers are anecdotes, not measurements; 30+ articles of each kind is a fair test.")
    disagree = (np.sign(scores.to_numpy()) != np.sign(scores.to_numpy()[:, [0]])).any(axis=1).sum()
    print(f"\nThe three models disagreed on {int(disagree)} of {len(df)} articles.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isot-fake", type=Path)
    ap.add_argument("--isot-true", type=Path)
    ap.add_argument("--welfake", type=Path)
    ap.add_argument("--welfake-flip", action="store_true", help="invert WELFake labels (needed for the usual Kaggle copy)")
    ap.add_argument("--models-dir", type=Path, default=Path("practical_models"))
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--text")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--explain", action="store_true", help="show the words driving each prediction")
    ap.add_argument("--explain-model", default="Both", choices=MODEL_NAMES)
    args = ap.parse_args()

    models = get_models(args)
    df = collect(args)
    if len(df):
        report(models, df, args)
    if args.interactive:
        while True:
            text = read_interactive()
            if text is None:
                break
            report(models, pd.DataFrame([dict(source="pasted", text=text, label=float("nan"))]), args)
    if not len(df) and not args.interactive:
        print("\nModels are ready. Now give it articles: --folder, --csv, --text or --interactive.")


if __name__ == "__main__":
    main()
