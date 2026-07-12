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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "delaked_data.csv"
MODELS_DIR = PROJECT_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        print("training_expectation: CPU-only; expect LSTM to take several minutes per epoch, possibly longer depending on sequence length and batch size")


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def build_vocab(texts: Iterable[str], max_vocab_size: int = 20000) -> tuple[dict[str, int], int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in most_common:
        vocab[token] = len(vocab)
    return vocab, len(vocab)


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> List[int]:
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids

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
        packed_out, (hidden, _) = self.lstm(packed)
        hidden = hidden.view(self.lstm.num_layers, 2, input_ids.size(0), self.lstm.hidden_size)
        forward_last = hidden[-1, 0]
        backward_last = hidden[-1, 1]
        features = torch.cat([forward_last, backward_last], dim=1)
        features = self.dropout(features)
        return self.classifier(features)


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(input_ids, lengths)
            batch_preds = torch.argmax(logits, dim=1)
            preds.extend(batch_preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return targets, preds


def train_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    start = time.perf_counter()
    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - start


def main() -> None:
    hardware_report()

    df = pd.read_csv(INPUT_PATH)
    print("\n===== Data =====")
    print(f"Loaded {INPUT_PATH.name} shape: {df.shape}")
    print(f"Loaded columns: {df.columns.tolist()}")
    expected_cols = {"text", "label", "clean_text"}
    if not expected_cols.issubset(df.columns):
        raise SystemExit(f"Required columns missing. Expected at least {sorted(expected_cols)}")

    text_col = "clean_text"
    y = df["label"].astype(int)
    X = df[text_col].fillna("").astype(str)

    original_rows = len(df)
    reduced_rows = 10000
    if reduced_rows < original_rows:
        sampled_parts = []
        for label_value, part in df.groupby("label"):
            target_n = max(1, round(reduced_rows * len(part) / original_rows))
            sampled_parts.append(part.sample(n=target_n, random_state=42))
        sampled = pd.concat(sampled_parts, ignore_index=True)
        sampled = sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df = sampled
        print(f"Dataset reduction applied for CPU practicality: {original_rows} -> {len(df)} rows (stratified sample)")
    else:
        print("No dataset reduction applied.")

    y = df["label"].astype(int)
    X = df[text_col].fillna("").astype(str)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X.tolist(),
        y.tolist(),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train/test sizes: {len(X_train_texts)} / {len(X_test_texts)}")
    print("Train class balance:")
    print(pd.Series(y_train).value_counts().sort_index().to_string())
    print("Test class balance:")
    print(pd.Series(y_test).value_counts().sort_index().to_string())

    vocab, vocab_size = build_vocab(X_train_texts, max_vocab_size=20000)
    print(f"Vocabulary cap: 20000")
    print(f"Actual vocab size (including special tokens): {vocab_size}")

    train_lengths = pd.Series([len(tokenize(text)) for text in X_train_texts])
    percentiles = train_lengths.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    print("Training text length percentiles (tokens):")
    for pct, value in percentiles.items():
        print(f"  p{int(pct * 100)}: {int(value)}")

    planned_max_len = 256
    max_len = 64
    print(f"Planned max sequence length: {planned_max_len}")
    print(f"Chosen max sequence length: {max_len}")
    print("Rationale: reduced from 256 to 64 for CPU practicality after the initial full-data run proved too slow; 64 keeps the BiLSTM tractable while still capturing the core content words.")

    def encode_split(texts: List[str], labels: List[int]):
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
    train_ds = TensorDataset(train_ids, train_len_tensor, train_label_tensor)
    test_ds = TensorDataset(test_ids, test_len_tensor, test_label_tensor)

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    embedding_dim = 32
    hidden_units = 32
    num_layers = 1
    dropout = 0.2
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    print("===== Architecture =====")
    print(f"device: {DEVICE}")
    print("original_planned_embedding_dim: 128")
    print("original_planned_hidden_units: 128")
    print("original_planned_dropout: 0.3")
    print(f"embedding_dim: {embedding_dim}")
    print(f"lstm_hidden_units: {hidden_units}")
    print(f"lstm_layers: {num_layers}")
    print(f"dropout: {dropout}")
    print(f"bidirectional: True")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs_requested = 1
    per_epoch_limit_seconds = 300
    trained_epochs = 0
    epoch_times: list[float] = []

    for epoch in range(1, epochs_requested + 1):
        print(f"\n===== Epoch {epoch}/{epochs_requested} =====")
        epoch_time = train_epoch(model, train_loader, optimizer, criterion)
        epoch_times.append(epoch_time)
        trained_epochs += 1
        print(f"epoch_time_seconds: {epoch_time:.2f}")
        if not torch.cuda.is_available() and epoch_time > per_epoch_limit_seconds:
            print(f"CPU-only guard triggered: epoch exceeded {per_epoch_limit_seconds} seconds. Stopping after 1 epoch and asking whether to continue.")
            break

    y_true, y_pred = evaluate(model, test_loader)
    print("\n===== Test Evaluation =====")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

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
                "max_len": max_len,
                "embedding_dim": embedding_dim,
                "hidden_units": hidden_units,
                "num_layers": num_layers,
                "dropout": dropout,
                "trained_epochs": trained_epochs,
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "device": str(DEVICE),
                "epoch_times_seconds": epoch_times,
            },
            f,
            indent=2,
        )

    print("\n===== Saved Artifacts =====")
    for path in [model_path, vocab_path, meta_path]:
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
