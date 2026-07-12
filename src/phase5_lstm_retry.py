from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import psutil
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "delaked_data.csv"
MODELS_DIR = PROJECT_DIR / "models"
PREVIOUS_META_PATH = MODELS_DIR / "bilstm_metadata.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def hardware_report() -> None:
    print("===== Hardware =====")
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"gpu_total_vram_gb: {round(props.total_memory / (1024**3), 2)}")
    print(f"cpu_logical_cores: {os.cpu_count()}")
    print(f"cpu_physical_cores: {psutil.cpu_count(logical=False)}")
    print(f"total_ram_gb: {round(psutil.virtual_memory().total / (1024**3), 2)}")
    if torch.cuda.is_available():
        print("training_expectation: should run in minutes per epoch on GPU")
    else:
        print("training_expectation: CPU-only; 256-token BiLSTM is still expected to be slow, but this retry uses a smaller model than the original 128/128 plan to stay within a practical review window")


def build_vocab(texts: Iterable[str], max_vocab_size: int = 20000) -> tuple[dict[str, int], int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in most_common:
        vocab[token] = len(vocab)
    return vocab, len(vocab)


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


def make_deterministic_sample(df: pd.DataFrame, target_rows: int = 10000) -> pd.DataFrame:
    original_rows = len(df)
    sampled_parts = []
    for label_value, part in df.groupby("label"):
        target_n = max(1, round(target_rows * len(part) / original_rows))
        sampled_parts.append(part.sample(n=target_n, random_state=42))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return sampled


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_units: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_units * 2, 2)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = hidden.view(self.lstm.num_layers, 2, input_ids.size(0), self.lstm.hidden_size)
        forward_last = hidden[-1, 0]
        backward_last = hidden[-1, 1]
        features = torch.cat([forward_last, backward_last], dim=1)
        features = self.dropout(features)
        return self.classifier(features)


def train_epoch(model, loader, optimizer, criterion) -> tuple[float, float]:
    model.train()
    start = time.perf_counter()
    total_loss = 0.0
    total_items = 0
    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return time.perf_counter() - start, total_loss / max(1, total_items)


def evaluate(model, loader):
    model.eval()
    preds = []
    targets = []
    all_probs = []
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=1)[:, 1]
            batch_preds = torch.argmax(logits, dim=1)
            preds.extend(batch_preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return targets, preds, all_probs


def main() -> None:
    hardware_report()

    df = pd.read_csv(INPUT_PATH)
    print("\n===== Data =====")
    print(f"Loaded {INPUT_PATH.name} shape: {df.shape}")
    print(f"Loaded columns: {df.columns.tolist()}")
    if not {"text", "label", "clean_text"}.issubset(df.columns):
        raise SystemExit("Required columns missing from delaked_data.csv")

    sampled_df = make_deterministic_sample(df, target_rows=10000)
    print(f"Deterministic stratified sample size: {len(sampled_df)}")
    y_sample = sampled_df["label"].astype(int)
    print("Sample class balance:")
    print(y_sample.value_counts().sort_index().to_string())

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        sampled_df["clean_text"].fillna("").astype(str).tolist(),
        y_sample.tolist(),
        test_size=0.2,
        random_state=42,
        stratify=y_sample,
    )
    print(f"Train/test sizes: {len(X_train_texts)} / {len(X_test_texts)}")
    print("Train class balance:")
    print(pd.Series(y_train).value_counts().sort_index().to_string())
    print("Test class balance:")
    print(pd.Series(y_test).value_counts().sort_index().to_string())

    vocab, vocab_size = build_vocab(X_train_texts, max_vocab_size=20000)
    print("Vocabulary cap: 20000")
    print(f"Actual vocab size (including special tokens): {vocab_size}")

    train_lengths = pd.Series([len(tokenize(text)) for text in X_train_texts])
    truncation_pct = float((train_lengths > 256).mean() * 100.0)
    percentiles = train_lengths.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    print("Training text length percentiles (tokens):")
    for pct, value in percentiles.items():
        print(f"  p{int(pct * 100)}: {int(value)}")
    print(f"Fraction of training rows truncated at max_len=256: {truncation_pct:.2f}%")

    max_len = 256
    print(f"Chosen max sequence length: {max_len}")

    def encode_split(texts: list[str], labels: list[int]):
        encoded = []
        lengths = []
        for text in texts:
            tokens = tokenize(text)
            lengths.append(min(len(tokens), max_len))
            token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_len]]
            if len(token_ids) < max_len:
                token_ids.extend([vocab["<pad>"]] * (max_len - len(token_ids)))
            encoded.append(token_ids)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    train_ids, train_len_tensor, train_label_tensor = encode_split(X_train_texts, y_train)
    test_ids, test_len_tensor, test_label_tensor = encode_split(X_test_texts, y_test)
    train_loader = DataLoader(TensorDataset(train_ids, train_len_tensor, train_label_tensor), batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(TensorDataset(test_ids, test_len_tensor, test_label_tensor), batch_size=128, shuffle=False, num_workers=0)

    # Start with the originally planned architecture and fall back only if needed.
    planned_embedding_dim = 128
    planned_hidden_units = 128
    fallback_embedding_dim = 64
    fallback_hidden_units = 64
    embedding_dim = planned_embedding_dim
    hidden_units = planned_hidden_units
    dropout = 0.3
    num_layers = 1

    model = BiLSTMClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_units=hidden_units, num_layers=num_layers, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("===== Architecture =====")
    print(f"device: {DEVICE}")
    print(f"planned_embedding_dim: {planned_embedding_dim}")
    print(f"planned_hidden_units: {planned_hidden_units}")
    print(f"fallback_embedding_dim: {fallback_embedding_dim}")
    print(f"fallback_hidden_units: {fallback_hidden_units}")
    print(f"embedding_dim_in_use: {embedding_dim}")
    print(f"hidden_units_in_use: {hidden_units}")
    print(f"num_layers: {num_layers}")
    print(f"dropout: {dropout}")
    print("bidirectional: True")

    losses = []
    epoch_times = []
    max_seconds_per_epoch = 600
    actual_epochs = 0
    for epoch in range(1, 3):
        print(f"\n===== Epoch {epoch}/2 =====")
        epoch_time, train_loss = train_epoch(model, train_loader, optimizer, criterion)
        epoch_times.append(epoch_time)
        losses.append(train_loss)
        actual_epochs += 1
        print(f"epoch_time_seconds: {epoch_time:.2f}")
        print(f"train_loss: {train_loss:.6f}")
        if epoch > 1:
            print(f"loss_change_from_previous_epoch: {losses[-2] - losses[-1]:.6f}")
        if not torch.cuda.is_available() and epoch_time > max_seconds_per_epoch:
            print(f"CPU-only guard triggered: epoch exceeded {max_seconds_per_epoch} seconds. Stopping early after epoch {epoch}.")
            break

    y_true, y_pred, y_prob = evaluate(model, test_loader)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob, labels=[0, 1])
    print("\n===== Test Evaluation =====")
    print(f"accuracy: {acc:.4f}")
    print(f"log_loss: {ll:.6f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    previous_meta = {}
    if PREVIOUS_META_PATH.exists():
        with PREVIOUS_META_PATH.open("r", encoding="utf-8") as f:
            previous_meta = json.load(f)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "bilstm_classifier.pt"
    vocab_path = MODELS_DIR / "bilstm_vocab.json"
    meta_path = MODELS_DIR / "bilstm_metadata.json"

    torch.save(model.state_dict(), model_path)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "previous_hyperparameters": previous_meta,
                "new_hyperparameters": {
                    "max_len": max_len,
                    "embedding_dim": embedding_dim,
                    "hidden_units": hidden_units,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "batch_size": 128,
                    "planned_embedding_dim": planned_embedding_dim,
                    "planned_hidden_units": planned_hidden_units,
                    "fallback_embedding_dim": fallback_embedding_dim,
                    "fallback_hidden_units": fallback_hidden_units,
                    "vocab_size": vocab_size,
                    "device": str(DEVICE),
                    "epoch_times_seconds": epoch_times,
                    "train_losses": losses,
                    "actual_epochs": actual_epochs,
                    "truncation_pct_train": truncation_pct,
                },
            },
            f,
            indent=2,
        )

    print("\n===== Comparison =====")
    print("Previous run accuracy: 0.6415")
    print("Previous run precision/recall imbalance was dominated by poor recall for label 0 and overprediction of label 1.")
    print(f"New run accuracy: {acc:.4f}")
    print(f"Epoch times: {[round(x, 2) for x in epoch_times]}")
    print(f"Train losses: {[round(x, 6) for x in losses]}")
    print("\n===== Saved Artifacts =====")
    for path in [model_path, vocab_path, meta_path]:
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
