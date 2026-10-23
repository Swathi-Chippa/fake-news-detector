"""
phase8_cross_dataset_dl.py
==========================
Deep-learning version of the cross-dataset experiment with SEVERAL model families.
Each model is trained on one dataset and tested on the datasets it has never seen.
It imports phase7_cross_dataset.py, so the data handling is identical to the classical run:

    * same loaders, same label convention (1 = True/real, 0 = Fake)
    * articles shared between datasets are removed BEFORE any split
    * same 80/20 in-domain split per seed, same metrics (accuracy, macro-F1,
      balanced accuracy, majority baseline, share predicted "True")

Models (choose with --models, comma-separated)
----------------------------------------------
  hf:<name>            any pretrained Hugging Face encoder, e.g.
                       hf:distilbert-base-uncased   (66M params, default)
                       hf:bert-base-uncased         (110M)
                       hf:roberta-base              (125M, about 2.5x slower than DistilBERT)
  bilstm               bidirectional LSTM trained from scratch (embedding 128, hidden 128)
  textcnn              CNN with filter widths 3/4/5 trained from scratch (Kim 2014 style)
  compact_transformer  small transformer encoder trained from scratch, INSPIRED by the
                       reference paper (sinusoidal positions, GELU, label smoothing 0.1).
                       It is NOT an exact replication: the paper's full spec and training
                       setup are not reproduced here.

The from-scratch models use the bert-base-uncased tokenizer for their vocabulary and
start with randomly initialised embeddings (no pretraining). That difference between
"pretrained" and "from scratch" is itself part of what the experiment tests.

Two text modes (the DL counterpart of phase7's "basic" / "delaked"):
    raw      = the article exactly as written (the model can see "(Reuters)", URLs, ...)
    delaked  = URLs and leakage terms removed, normal words and punctuation kept
    strict   = delaked + dateline removed + ALL punctuation removed (letters/digits only)

Needs a GPU for practical speed (Kaggle: Settings -> Accelerator -> GPU) and internet once
(pretrained weights + tokenizer download). Both scripts must sit in the same folder:

    python phase8_cross_dataset_dl.py --isot-fake Fake.csv --isot-true True.csv \\
        --welfake WELFake_Dataset.csv --welfake-flip --liar-csv liar_binary.csv

Quick smoke run (a few minutes):
    ... --max-train 3000 --max-eval 3000 --epochs 1 --mode delaked
"""
from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import phase7_cross_dataset as p7

CHAR_LIMIT = 2000                      # 256 tokens is roughly 1,500 characters
SCRATCH_TOKENIZER = "bert-base-uncased"
DEFAULT_MODELS = "hf:distilbert-base-uncased,bilstm,textcnn,compact_transformer"

# (epochs, learning rate) used when --epochs / --lr are not given
HPARAMS = {
    "hf": (2, 2e-5),
    "bilstm": (4, 1e-3),
    "textcnn": (4, 1e-3),
    "compact_transformer": (5, 5e-4),
}


def kind_of(name: str) -> str:
    return "hf" if name.startswith("hf:") else name


def short_name(name: str) -> str:
    return name[3:] if name.startswith("hf:") else name


def hparams(name: str, args) -> tuple[int, float]:
    if kind_of(name) not in HPARAMS:
        raise SystemExit(f"Unknown model '{name}'. Use hf:<name>, bilstm, textcnn or compact_transformer.")
    epochs, lr = HPARAMS[kind_of(name)]
    return (args.epochs or epochs), (args.lr or lr)


# --------------------------------------------------------------------------
# Text preparation
# --------------------------------------------------------------------------
# "CITY (Reuters) -" at the start of an article, e.g. "WASHINGTON (Reuters) - ..."
DATELINE = re.compile(r"^\s*[A-Z][A-Za-z .,'/&-]{1,60}?\s*\((?:Reuters|AP|AFP)[^)]{0,20}\)\s*[-\u2013\u2014]*\s*")


def prepare_text(text: str, mode: str) -> str:
    text = str(text)[:CHAR_LIMIT]
    if mode == "raw":
        return text
    if mode == "delaked":
        # phase7 cleaning only; punctuation is kept, so formatting leftovers such as "( ) -" survive
        return p7.remove_leakage_terms(p7.strip_urls(text))
    # "strict": also drop the dateline and ALL punctuation / symbols / typographic quotes,
    # keeping only lowercase letters, digits and spaces (stop-words stay, unlike phase7's classical cleaning)
    text = DATELINE.sub("", text)
    cleaned = p7.remove_leakage_terms(p7.strip_urls(text))
    cleaned = re.sub(r"\b21st century\b", " ", cleaned)           # remaining outlet name in ISOT's fake articles
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def cap(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Stratified random subsample to about n rows (keeps the class balance)."""
    if not n or len(df) <= n:
        return df
    return df.groupby("label", group_keys=False).sample(frac=n / len(df), random_state=seed)


# --------------------------------------------------------------------------
# Torch / transformers parts (imported lazily so the rest can be tested without a GPU)
# --------------------------------------------------------------------------
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_TOKENIZERS: dict = {}


def get_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    key = model_name[3:] if model_name.startswith("hf:") else SCRATCH_TOKENIZER
    if key not in _TOKENIZERS:
        _TOKENIZERS[key] = AutoTokenizer.from_pretrained(key)
    return _TOKENIZERS[key]


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_SCRATCH: dict = {}


def _scratch_classes() -> dict:
    """Models trained from scratch. Each forward() mimics Hugging Face: returns .loss and .logits."""
    if _SCRATCH:
        return _SCRATCH
    import math
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    def finish(logits, labels, smoothing: float = 0.0):
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.float(), labels, label_smoothing=smoothing)
        return SimpleNamespace(loss=loss, logits=logits)

    def masked_mean(h, attention_mask):
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    class BiLSTM(nn.Module):
        def __init__(self, vocab, pad_id, emb=128, hid=128):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
            self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(2 * hid, 2)

        def forward(self, input_ids, attention_mask=None, labels=None, **kw):
            lengths = attention_mask.sum(1).clamp(min=1).cpu()
            x = self.drop(self.emb(input_ids))
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=input_ids.size(1))
            return finish(self.out(self.drop(masked_mean(h, attention_mask))), labels)

    class TextCNN(nn.Module):
        def __init__(self, vocab, pad_id, emb=128, filters=100, widths=(3, 4, 5)):
            super().__init__()
            self.widths = widths
            self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
            self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k) for k in widths])
            self.drop = nn.Dropout(0.5)
            self.out = nn.Linear(filters * len(widths), 2)

        def forward(self, input_ids, attention_mask=None, labels=None, **kw):
            x = self.emb(input_ids).transpose(1, 2)                # batch, emb, length
            if x.size(2) < max(self.widths):                       # very short batches
                x = nn.functional.pad(x, (0, max(self.widths) - x.size(2)))
            feats = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
            return finish(self.out(self.drop(torch.cat(feats, dim=1))), labels)

    class CompactTransformer(nn.Module):
        def __init__(self, vocab, pad_id, d=192, heads=6, layers=4, ff=384, max_len=512, drop=0.1):
            super().__init__()
            self.d = d
            self.emb = nn.Embedding(vocab, d, padding_idx=pad_id)
            pe = torch.zeros(max_len, d)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe)
            layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation="gelu", batch_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)
            self.out = nn.Linear(d, 2)

        def forward(self, input_ids, attention_mask=None, labels=None, **kw):
            x = self.emb(input_ids) * math.sqrt(self.d) + self.pe[: input_ids.size(1)]
            h = self.enc(self.drop(x), src_key_padding_mask=(attention_mask == 0))
            pooled = self.norm(masked_mean(h, attention_mask))
            return finish(self.out(self.drop(pooled)), labels, smoothing=0.1)

    _SCRATCH.update(bilstm=BiLSTM, textcnn=TextCNN, compact_transformer=CompactTransformer)
    return _SCRATCH


def build_model(name: str, tok):
    if name.startswith("hf:"):
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(name[3:], num_labels=2)
    classes = _scratch_classes()
    if name not in classes:
        raise SystemExit(f"Unknown model '{name}'.")
    return classes[name](len(tok), tok.pad_token_id)


def train_model(model_name: str, texts: list[str], labels: list[int], tok, device, args, seed: int):
    """Train a fresh model on (texts, labels). Returns the trained model."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    epochs, lr = hparams(model_name, args)
    set_seed(seed)
    use_amp = device.type == "cuda"
    model = build_model(model_name, tok).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    {short_name(model_name)}: {n_params:.1f}M parameters, {epochs} epochs, lr {lr}", flush=True)

    def collate(batch_idx):
        enc = tok([texts[i] for i in batch_idx], truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in batch_idx])
        return enc

    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(list(range(len(texts))), batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=gen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    step, t0, running = 0, time.time(), 0.0
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"    epoch {epoch+1}/{epochs} step {step}/{total_steps} "
                      f"loss {running/100:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                running = 0.0
    return model


def predict(model, tok, texts: list[str], device, args) -> np.ndarray:
    """Predicted labels (0/1) for texts, in the original order."""
    import torch

    use_amp = device.type == "cuda"
    model.eval()
    order = np.argsort([len(t) for t in texts])          # similar lengths together = less padding
    preds = np.zeros(len(texts), dtype=int)
    with torch.no_grad():
        for i in range(0, len(order), args.eval_batch_size):
            ids = order[i:i + args.eval_batch_size]
            enc = tok([texts[j] for j in ids], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            preds[ids] = logits.argmax(dim=-1).cpu().numpy()
    return preds


def free_model(model) -> None:
    try:
        import gc
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isot-fake", type=Path)
    ap.add_argument("--isot-true", type=Path)
    ap.add_argument("--welfake", type=Path)
    ap.add_argument("--welfake-flip", action="store_true")
    ap.add_argument("--liar-dir", type=Path)
    ap.add_argument("--liar-csv", type=Path)
    ap.add_argument("--no-dedup-across", action="store_true")
    ap.add_argument("--mode", choices=["raw", "delaked", "strict", "both", "all"], default="both",
                    help="both = raw + delaked; all = raw + delaked + strict")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated, see top of file")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for ALL models")
    ap.add_argument("--lr", type=float, default=None, help="override learning rate for ALL models")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=20000, help="cap on training rows per run")
    ap.add_argument("--max-eval", type=int, default=20000, help="cap on rows per test set")
    ap.add_argument("--max-per-dataset", type=int, default=None, help="cap each dataset before everything else")
    ap.add_argument("--seeds", default="42", help="comma-separated; each extra seed multiplies the run time")
    ap.add_argument("--out-dir", type=Path, default=Path("cross_dataset_results_dl"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_names:
        hparams(m, args)                                  # fail early on a typo
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load + dedup exactly like phase7 -------------------------------------------------
    raw: dict[str, pd.DataFrame] = {}
    if args.isot_fake and args.isot_true:
        raw["ISOT"] = p7.finalize(p7.load_isot(args.isot_fake, args.isot_true), "isot")
    if args.welfake:
        raw["WELFake"] = p7.finalize(p7.load_welfake(args.welfake, args.welfake_flip), "welfake")
    if args.liar_dir:
        raw["LIAR"] = p7.finalize(p7.load_liar(args.liar_dir), "liar")
    elif args.liar_csv:
        raw["LIAR"] = p7.finalize(p7.load_liar_csv(args.liar_csv), "liar")
    if len(raw) < 2:
        raise SystemExit("Need at least 2 datasets.")
    if not args.no_dedup_across:
        print("\n=== Removing articles shared between datasets (before any split) ===")
        p7.dedup_across(raw)
    raw = {n: p7.subsample(d, args.max_per_dataset) for n, d in raw.items()}

    print("\n=== Datasets ===")
    for n, d in raw.items():
        print(f"{n:8s} rows={len(d):6d}  true={d['label'].mean()*100:5.1f}%")

    device = get_device()
    print(f"\ndevice: {getattr(device, 'type', device)} | models: {model_names} | max_len: {args.max_len} "
          f"| max_train: {args.max_train} | seeds: {seeds}")

    names = list(raw)
    modes = {"both": ["raw", "delaked"], "all": ["raw", "delaked", "strict"]}.get(args.mode, [args.mode])
    all_rows: list[dict] = []

    for mode in modes:
        t0 = time.time()
        prepared = {}
        for n, d in raw.items():
            d = d.copy()
            d["x"] = d["text"].map(lambda t: prepare_text(t, mode))
            prepared[n] = d[d["x"].str.len() > 0].reset_index(drop=True)
        print(f"\n=== Text prepared ({mode}) in {time.time()-t0:.0f}s ===")

        mode_rows: list[dict] = []
        for model_name in model_names:
            label = short_name(model_name)
            tok = get_tokenizer(model_name)
            for src in names:
                df = prepared[src]
                for seed in seeds:
                    tr_idx, te_idx = train_test_split(
                        np.arange(len(df)), test_size=0.2, random_state=seed, stratify=df["label"])
                    train = cap(df.iloc[tr_idx], args.max_train, seed)
                    own_test = df.iloc[te_idx]
                    print(f"\n[{mode}] {label}: training on {src} (seed {seed}), {len(train)} articles", flush=True)
                    model = train_model(model_name, train["x"].tolist(), train["label"].astype(int).tolist(),
                                        tok, device, args, seed)
                    for tgt in names:
                        test = own_test if tgt == src else prepared[tgt]
                        test = cap(test, args.max_eval, 0)
                        pred = predict(model, tok, test["x"].tolist(), device, args)
                        y = test["label"].to_numpy().astype(int)
                        row = dict(
                            mode=mode, train=src, test=tgt, model=label, seed=seed, n_test=len(test),
                            accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred),
                            macro_f1=f1_score(y, pred, average="macro"),
                            pred_true_rate=float(pred.mean()), actual_true_rate=float(y.mean()),
                            majority_baseline=float(max(y.mean(), 1 - y.mean())),
                        )
                        mode_rows.append(row)
                        print(f"    -> tested on {tgt:8s} n={len(test):6d} acc={row['accuracy']*100:5.2f} "
                              f"bal_acc={row['balanced_acc']*100:5.2f} predicted-true={row['pred_true_rate']*100:5.1f}%",
                              flush=True)
                    free_model(model)
                    pd.DataFrame(all_rows + mode_rows).to_csv(args.out_dir / "results_dl_partial.csv", index=False)

        res = pd.DataFrame(mode_rows)
        res.to_csv(args.out_dir / f"results_dl_{mode}.csv", index=False)
        all_rows.extend(mode_rows)

        for model_name in model_names:
            label = short_name(model_name)
            acc_m, acc_s = p7.pivot(res, label, "accuracy", names)
            f1_m, f1_s = p7.pivot(res, label, "macro_f1", names)
            bal_m, bal_s = p7.pivot(res, label, "balanced_acc", names)
            print(f"\n--- {mode} | {label} | accuracy % (mean +- std over {len(seeds)} seed(s)) ---")
            print(p7.fmt_table(acc_m, acc_s).to_string())
            print(f"--- {mode} | {label} | macro-F1 % ---")
            print(p7.fmt_table(f1_m, f1_s).to_string())
            print(f"--- {mode} | {label} | balanced accuracy % (50 = constant guess) ---")
            print(p7.fmt_table(bal_m, bal_s).to_string())
            p7.heatmap(acc_m, f"{label} ({mode} text)", args.out_dir / f"matrix_dl_{mode}_{label.replace('/', '_')}.png")

        diag = (res[res["train"] != res["test"]]
                .groupby(["model", "train", "test"])[["accuracy", "balanced_acc", "majority_baseline",
                                                      "pred_true_rate", "actual_true_rate"]].mean().round(3))
        print(f"\n--- {mode} | cross-dataset diagnostics (all models) ---")
        print(diag.to_string())

    full = pd.DataFrame(all_rows)
    full.to_csv(args.out_dir / "results_dl_all.csv", index=False)

    print("\n=== Summary: ISOT <-> WELFake only (the two news-article datasets), accuracy % ===")
    news = full[full["train"].isin(["ISOT", "WELFake"]) & full["test"].isin(["ISOT", "WELFake"])]
    if not news.empty:
        news = news.assign(kind=np.where(news["train"] == news["test"], "in-domain", "cross-dataset"))
        summ = news.groupby(["mode", "model", "kind"])["accuracy"].mean().unstack("kind") * 100
        if {"in-domain", "cross-dataset"} <= set(summ.columns):
            summ["gap"] = summ["in-domain"] - summ["cross-dataset"]
        print(summ.round(2).to_string())
    print(f"\nSaved CSVs and heatmaps to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
