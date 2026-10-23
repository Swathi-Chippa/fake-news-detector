"""
phase7_cross_dataset.py
=======================
Cross-dataset generalization experiment for the Fake News Detector project.

Question answered: "I trained on ISOT and got ~97%. What happens when the SAME
pipeline (TF-IDF 1-2 grams / 5000 features -> LinearSVC etc.) is trained on one
dataset and tested on a dataset it has never seen?"

It reuses your final pipeline from phase6_rebuild_corrected_v2.py
(strip URLs -> remove leakage terms -> standard clean) and compares it against
the basic phase2 cleaning, so you also get a leakage ablation for free:

    mode "basic"   = phase2 cleaning only (lowercase, letters only, stopwords)
    mode "delaked" = your final pipeline (URLs + Reuters/wire/day/month terms removed)

Outputs (in --out-dir):
    results_<mode>.csv            long-form table, every (train, test, model, seed)
    matrix_<mode>_<model>.png     heatmap, rows = trained on, cols = tested on
    (and printed accuracy / F1 tables with mean +- std over seeds)

Usage examples
--------------
# ISOT + WELFake + LIAR (any subset of >= 2 datasets works)
python phase7_cross_dataset.py \
    --isot-fake Fake.csv --isot-true True.csv \
    --welfake WELFake_Dataset.csv \
    --liar-dir liar_dataset/          # folder holding train.tsv valid.tsv test.tsv

# quick smoke run on a subset
python phase7_cross_dataset.py ... --max-per-dataset 5000 --seeds 42,43

Label convention (same as your project): 1 = True/real, 0 = Fake.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------
# Cleaning (copied from your phase6_rebuild_corrected_v2.py so results match)
# --------------------------------------------------------------------------
LEAKAGE_PHRASES = [
    "washington reuters", "told reuters", "told reporters", "featured image",
    "image via", "getty images", "twitter com", "daily mail", "reuters", "said",
    "via", "read", "featured", "image", "getty", "images", "com", "wire",
    "watch", "ap", "us", "gop", "hillary", "mr", "obama", "rep", "sen",
]
DAY_WORDS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_WORDS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+\b|\b[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?\b"
)

_PHRASE_PATTERNS = [
    re.compile(rf"\b{re.escape(p)}\b")
    for p in sorted(set(LEAKAGE_PHRASES), key=len, reverse=True)
]
_DAY_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, DAY_WORDS)) + r")\b")
_MONTH_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, MONTH_WORDS)) + r")\b")


def _load_stop_words() -> set[str]:
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except Exception:  # offline / nltk missing
        print("[warn] NLTK stopwords unavailable, falling back to sklearn's stop-word list "
              "(numbers will differ very slightly from your README).")
        return set(ENGLISH_STOP_WORDS)


STOP_WORDS = _load_stop_words()


def strip_urls(text: str) -> str:
    cleaned = str(text).lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\b(?:http|https|www|youtube|youtu|com|org|net|edu|gov|bitly|bit\.ly)\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def remove_leakage_terms(text: str) -> str:
    cleaned = str(text).lower()
    for pat in _PHRASE_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = _DAY_PATTERN.sub(" ", cleaned)
    cleaned = _MONTH_PATTERN.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def standard_clean(text: str) -> str:
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = [t for t in cleaned.split()
              if t not in STOP_WORDS and not re.search(r"(http|www|youtube)", t)]
    return " ".join(tokens)


def basic_clean(text: str) -> str:
    """phase2: lowercase, letters only, stop-words removed."""
    cleaned = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join(t for t in cleaned.split() if t not in STOP_WORDS)


def delaked_clean(text: str) -> str:
    """Your final pipeline (phase6_rebuild_corrected_v2.rebuild_clean_text)."""
    return standard_clean(remove_leakage_terms(strip_urls(text)))


def real_word_count(text: str) -> int:
    cleaned = re.sub(r"[^a-z\s]", " ", strip_urls(text))
    return len(cleaned.split())


CLEANERS = {"basic": basic_clean, "delaked": delaked_clean}

# --------------------------------------------------------------------------
# Dataset loaders -> DataFrame[text, label]  (1 = True/real, 0 = Fake)
# --------------------------------------------------------------------------

def _read_csv(path: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kw)


def load_isot(fake_path: Path, true_path: Path) -> pd.DataFrame:
    fake = _read_csv(fake_path)
    true = _read_csv(true_path)
    fake["label"], true["label"] = 0, 1
    return pd.concat([true[["text", "label"]], fake[["text", "label"]]], ignore_index=True)


def load_welfake(path: Path, flip: bool) -> pd.DataFrame:
    df = _read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise SystemExit(f"WELFake file needs 'text' and 'label' columns, found {list(df.columns)}")
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype(int)
    # NOTE: WELFake's documentation is inconsistent about which value means 'fake'.
    # Default here: 1 = real, 0 = fake (matches the paper). Use --welfake-flip if
    # the sanity check at the end warns about systematically below-chance accuracy.
    if flip:
        df["label"] = 1 - df["label"]
    return df


LIAR_COLS = ["id", "label", "text", "subject", "speaker", "job", "state", "party",
             "c_barely_true", "c_false", "c_half_true", "c_mostly_true", "c_pants_fire", "context"]
LIAR_MAP = {"true": 1, "mostly-true": 1, "false": 0, "barely-true": 0, "pants-fire": 0}  # half-true dropped


def load_liar(folder: Path) -> pd.DataFrame:
    frames = []
    for name in ("train.tsv", "valid.tsv", "test.tsv"):
        p = folder / name
        if p.exists():
            frames.append(pd.read_csv(p, sep="\t", header=None, names=LIAR_COLS, quoting=3))
    if not frames:
        raise SystemExit(f"No train.tsv / valid.tsv / test.tsv found in {folder}")
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].map(LIAR_MAP)  # 'half-true' -> NaN -> dropped
    return df[["text", "label"]].dropna(subset=["label"]).astype({"label": int})


def load_liar_csv(path: Path) -> pd.DataFrame:
    """Binary LIAR exported as a CSV with columns text,label (1 = true, 0 = false)."""
    df = _read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise SystemExit(f"--liar-csv needs 'text' and 'label' columns, found {list(df.columns)}")
    return df[["text", "label"]].dropna(subset=["label"]).astype({"label": int})


MIN_WORDS = {"isot": 20, "welfake": 20, "liar": 5}  # LIAR statements are one-liners


def finalize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df[df["text"].map(real_word_count) >= MIN_WORDS[name]].reset_index(drop=True)
    return df


def subsample(df: pd.DataFrame, max_n: int | None) -> pd.DataFrame:
    if not max_n or len(df) <= max_n:
        return df
    parts = [g.sample(int(round(max_n * len(g) / len(df))), random_state=0) for _, g in df.groupby("label")]
    return pd.concat(parts, ignore_index=True)


def dedup_across(raw: dict[str, pd.DataFrame]) -> None:
    """Remove from each dataset every article already present in an EARLIER dataset
    (priority = order loaded: ISOT, WELFake, LIAR). Done BEFORE any train/test split, so
    no article can sit in both the training side and the test side of a cross-dataset pair.
    Also prints how often the duplicated articles carry the same label in both datasets."""
    seen: dict[str, int] = {}
    for name in list(raw):
        df = raw[name]
        key = df["text"].map(basic_clean)
        dup = key.isin(seen)
        if dup.any():
            same = (df.loc[dup, "label"].to_numpy() == key[dup].map(seen).to_numpy()).mean()
            verdict = "labels consistent" if same > 0.9 else ("LABELS LOOK INVERTED -> try --welfake-flip" if same < 0.1 else "mixed")
            print(f"[dedup] {name}: {int(dup.sum())} of {len(df)} articles already in an earlier dataset "
                  f"({dup.mean()*100:.1f}%); their labels agree {same*100:.1f}% -> {verdict}")
        else:
            print(f"[dedup] {name}: no articles shared with earlier datasets")
        kept = df[~dup].reset_index(drop=True)
        for k, l in zip(key[~dup], kept["label"]):
            seen[k] = int(l)
        raw[name] = kept


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
MODEL_FACTORIES = {
    "linear_svc": lambda seed: LinearSVC(),
    "logreg": lambda seed: LogisticRegression(max_iter=1000),
    "naive_bayes": lambda seed: MultinomialNB(),
}


def run(datasets: dict[str, pd.DataFrame], mode: str, models: list[str], seeds: list[int]) -> pd.DataFrame:
    names = list(datasets)
    rows = []
    for src in names:
        df = datasets[src]
        for seed in seeds:
            tr_idx, te_idx = train_test_split(
                np.arange(len(df)), test_size=0.2, random_state=seed, stratify=df["label"])
            train, own_test = df.iloc[tr_idx], df.iloc[te_idx]
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)   # fit on TRAIN ONLY
            Xtr = vec.fit_transform(train["clean"])
            train_set = set(train["clean"])
            fitted = {m: MODEL_FACTORIES[m](seed).fit(Xtr, train["label"]) for m in models}

            for tgt in names:
                if tgt == src:
                    test = own_test
                    overlap_removed = 0
                    overlap_agree = float("nan")
                else:
                    test = datasets[tgt]
                    keep = ~test["clean"].isin(train_set)   # drop exact duplicates seen in training
                    overlap_removed = int((~keep).sum())
                    if overlap_removed:
                        # do the two datasets give the SAME article the same label?
                        lab = dict(zip(train["clean"], train["label"]))
                        dup = test[~keep]
                        overlap_agree = float((dup["label"].to_numpy() == dup["clean"].map(lab).to_numpy()).mean())
                    else:
                        overlap_agree = float("nan")
                    test = test[keep]
                if len(test) == 0:
                    continue
                Xte = vec.transform(test["clean"])
                y = test["label"].to_numpy()
                majority = max(y.mean(), 1 - y.mean())
                for m, clf in fitted.items():
                    pred = clf.predict(Xte)
                    rows.append(dict(
                        mode=mode, train=src, test=tgt, model=m, seed=seed, n_test=len(test),
                        accuracy=accuracy_score(y, pred),
                        balanced_acc=balanced_accuracy_score(y, pred),
                        macro_f1=f1_score(y, pred, average="macro"),
                        pred_true_rate=float(pred.mean()), actual_true_rate=float(y.mean()),
                        majority_baseline=float(majority), overlap_removed=overlap_removed,
                        overlap_label_agreement=overlap_agree,
                    ))
    return pd.DataFrame(rows)


def pivot(res: pd.DataFrame, model: str, metric: str, names: list[str]):
    sub = res[res["model"] == model]
    g = sub.groupby(["train", "test"])[metric].agg(["mean", "std"]).reset_index()
    mean = g.pivot(index="train", columns="test", values="mean").reindex(index=names, columns=names)
    std = g.pivot(index="train", columns="test", values="std").reindex(index=names, columns=names)
    return mean, std


def fmt_table(mean: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    out = mean.copy().astype(object)
    for r in mean.index:
        for c in mean.columns:
            m, s = mean.loc[r, c], std.loc[r, c]
            out.loc[r, c] = "-" if pd.isna(m) else f"{m*100:5.2f} +- {0 if pd.isna(s) else s*100:4.2f}"
    out.index.name = "trained on \\ tested on"
    return out


def heatmap(mean: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.8 + 1.5 * len(mean.columns), 1.6 + 1.2 * len(mean)))
    im = ax.imshow(mean.values * 100, vmin=40, vmax=100, cmap="RdYlGn")
    ax.set_xticks(range(len(mean.columns)), mean.columns)
    ax.set_yticks(range(len(mean.index)), mean.index)
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.set_title(title)
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            v = mean.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="accuracy (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isot-fake", type=Path)
    ap.add_argument("--isot-true", type=Path)
    ap.add_argument("--welfake", type=Path, help="WELFake_Dataset.csv")
    ap.add_argument("--welfake-flip", action="store_true", help="invert WELFake labels")
    ap.add_argument("--liar-dir", type=Path, help="folder with train.tsv / valid.tsv / test.tsv")
    ap.add_argument("--liar-csv", type=Path, help="binary LIAR as CSV with columns text,label (1=true)")
    ap.add_argument("--no-dedup-across", action="store_true",
                    help="keep articles that appear in more than one dataset (NOT recommended)")
    ap.add_argument("--mode", choices=["basic", "delaked", "both"], default="both")
    ap.add_argument("--models", default="linear_svc,logreg,naive_bayes")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--max-per-dataset", type=int, default=None, help="stratified cap for quick runs")
    ap.add_argument("--out-dir", type=Path, default=Path("cross_dataset_results"))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw: dict[str, pd.DataFrame] = {}
    if args.isot_fake and args.isot_true:
        raw["ISOT"] = finalize(load_isot(args.isot_fake, args.isot_true), "isot")
    if args.welfake:
        raw["WELFake"] = finalize(load_welfake(args.welfake, args.welfake_flip), "welfake")
    if args.liar_dir:
        raw["LIAR"] = finalize(load_liar(args.liar_dir), "liar")
    elif args.liar_csv:
        raw["LIAR"] = finalize(load_liar_csv(args.liar_csv), "liar")
    if len(raw) < 2:
        raise SystemExit("Need at least 2 datasets (e.g. --isot-fake/--isot-true plus --welfake or --liar-dir).")

    if not args.no_dedup_across:
        print("\n=== Removing articles shared between datasets (before any split) ===")
        dedup_across(raw)
    raw = {n: subsample(d, args.max_per_dataset) for n, d in raw.items()}

    print("\n=== Datasets after dedup / length filter ===")
    for n, d in raw.items():
        print(f"{n:8s} rows={len(d):6d}  true={d['label'].mean()*100:5.1f}%  "
              f"median words={int(d['text'].str.split().str.len().median())}")

    names = list(raw)
    modes = ["basic", "delaked"] if args.mode == "both" else [args.mode]
    all_res = []
    for mode in modes:
        t0 = time.time()
        print(f"\n=== Cleaning ({mode}) ===")
        prepared = {}
        for n, d in raw.items():
            d = d.copy()
            d["clean"] = d["text"].map(CLEANERS[mode])
            prepared[n] = d[d["clean"].str.len() > 0].reset_index(drop=True)
        print(f"cleaned in {time.time()-t0:.0f}s; training / evaluating ...")
        res = run(prepared, mode, models, seeds)
        res.to_csv(args.out_dir / f"results_{mode}.csv", index=False)
        all_res.append(res)

        for m in models:
            acc_m, acc_s = pivot(res, m, "accuracy", names)
            f1_m, f1_s = pivot(res, m, "macro_f1", names)
            print(f"\n--- {mode} | {m} | accuracy % (mean +- std over {len(seeds)} seeds) ---")
            print(fmt_table(acc_m, acc_s).to_string())
            print(f"--- {mode} | {m} | macro-F1 % ---")
            print(fmt_table(f1_m, f1_s).to_string())
            heatmap(acc_m, f"{m} ({mode} cleaning)", args.out_dir / f"matrix_{mode}_{m}.png")

        # diagnostic: what does the primary model predict on foreign data?
        primary = models[0]
        diag = (res[(res["model"] == primary) & (res["train"] != res["test"])]
                .groupby(["train", "test"])[["accuracy", "balanced_acc", "majority_baseline", "pred_true_rate",
                                             "actual_true_rate", "overlap_removed",
                                             "overlap_label_agreement"]].mean().round(3))
        print(f"\n--- {mode} | {primary} | cross-dataset diagnostics "
              f"(acc vs majority baseline; share predicted 'True' vs actual) ---")
        print(diag.to_string())

    full = pd.concat(all_res, ignore_index=True)
    primary = models[0]
    print("\n=== Summary for", primary, "===")
    for mode in modes:
        s = full[(full["mode"] == mode) & (full["model"] == primary)]
        within = s[s["train"] == s["test"]]["accuracy"].mean() * 100
        cross = s[s["train"] != s["test"]]["accuracy"].mean() * 100
        print(f"{mode:8s} in-domain {within:5.2f}%   cross-dataset {cross:5.2f}%   gap {within-cross:5.2f} pts")

    # sanity check for inverted labels. A constant predictor scores 0.50 balanced accuracy,
    # so values well below 0.5 (or overlapping articles that carry OPPOSITE labels) point to
    # inverted labels - not just class imbalance.
    x = full[(full["model"] == primary) & (full["train"] != full["test"])]
    low = x[x["balanced_acc"] < 0.35]
    if len(low):
        pairs = sorted({f"{a}->{b}" for a, b in zip(low["train"], low["test"])})
        print("\n[WARNING] Balanced accuracy far below 0.5 for:", ", ".join(pairs))
    agree = x.dropna(subset=["overlap_label_agreement"]).groupby(["train", "test"])["overlap_label_agreement"].mean()
    for (a, b), v in agree.items():
        verdict = "labels consistent" if v > 0.9 else ("LABELS LOOK INVERTED -> try --welfake-flip" if v < 0.1 else "mixed")
        print(f"[overlap check] same articles in {a} and {b}: labels agree {v*100:.1f}% -> {verdict}")

    print(f"\nSaved CSVs and heatmaps to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
