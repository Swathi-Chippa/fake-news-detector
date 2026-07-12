from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
except ImportError as exc:  # pragma: no cover - dependency bootstrap path
    raise SystemExit(
        "transformers is required for Phase 5.2. Install it with `python -m pip install transformers accelerate`."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "delaked_data.csv"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = MODELS_DIR / "distilbert_classifier"
MODEL_NAME = "distilbert-base-uncased"
TARGET_ROWS = 2000
TRAIN_ROWS = 1600
TEST_ROWS = 400
MAX_LENGTH = 96
INITIAL_BATCH_SIZE = 4
MIN_FREE_RAM_GB = 1.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ram_report() -> None:
    vm = psutil.virtual_memory()
    print("===== Hardware =====")
    print(f"device: {DEVICE}")
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"total_ram_gb: {vm.total / (1024**3):.2f}")
    print(f"available_ram_gb: {vm.available / (1024**3):.2f}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
        print(f"gpu_total_vram_gb: {props.total_memory / (1024**3):.2f}")


def sample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < TARGET_ROWS:
        raise SystemExit(f"delaked_data.csv has only {len(df)} rows, fewer than the requested {TARGET_ROWS}.")
    groups = list(df.groupby("label", sort=True))
    target_counts = {}
    remainders = []
    allocated = 0
    for label_value, group in groups:
        raw_target = TARGET_ROWS * len(group) / len(df)
        base = int(raw_target)
        target_counts[label_value] = base
        allocated += base
        remainders.append((raw_target - base, label_value))

    remaining = TARGET_ROWS - allocated
    for _, label_value in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        target_counts[label_value] += 1
        remaining -= 1

    sampled_parts = []
    for label_value, group in groups:
        sampled_parts.append(group.sample(n=target_counts[label_value], random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)


def encode_texts(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {key: value for key, value in encoded.items()}


def make_dataloaders(tokenizer, train_texts, train_labels, test_texts, test_labels, batch_size: int):
    train_encoded = encode_texts(tokenizer, train_texts)
    test_encoded = encode_texts(tokenizer, test_texts)

    train_ds = TensorDataset(
        train_encoded["input_ids"],
        train_encoded["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long),
    )
    test_ds = TensorDataset(
        test_encoded["input_ids"],
        test_encoded["attention_mask"],
        torch.tensor(test_labels, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024**3)


def print_ram_checkpoint(label: str) -> None:
    free_gb = available_ram_gb()
    print(f"ram_checkpoint[{label}]: {free_gb:.2f} GB free")
    return free_gb


def memory_guard(label: str) -> None:
    free_gb = print_ram_checkpoint(label)
    if free_gb < MIN_FREE_RAM_GB:
        raise SystemExit(
            f"Available RAM dropped to {free_gb:.2f} GB at checkpoint '{label}', below the {MIN_FREE_RAM_GB:.2f} GB safety threshold. Stopping immediately."
        )


def train_one_epoch(model, loader, optimizer) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_items = 0
    batch_count = 0
    start = time.perf_counter()
    report_batch = max(1, len(loader) // 10)

    for batch_idx, batch in enumerate(loader, start=1):
        batch_label = f"train_batch_{batch_idx}"
        should_report = batch_idx == 1 or batch_idx == len(loader) or batch_idx % report_batch == 0
        if should_report:
            memory_guard(batch_label)
        elif available_ram_gb() < MIN_FREE_RAM_GB:
            raise SystemExit(
                f"Available RAM dropped below {MIN_FREE_RAM_GB:.2f} GB during training at checkpoint '{batch_label}'. Stopping immediately."
            )

        input_ids, attention_mask, labels = [tensor.to(DEVICE) for tensor in batch]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        batch_count += 1

        if batch_idx == report_batch:
            elapsed = time.perf_counter() - start
            avg_batch_seconds = elapsed / batch_idx
            projected_total_seconds = avg_batch_seconds * len(loader)
            print("===== Early Time Estimate =====")
            print(f"batches_completed: {batch_idx}/{len(loader)}")
            print(f"elapsed_seconds: {elapsed:.2f}")
            print(f"projected_total_seconds: {projected_total_seconds:.2f}")
            print(f"projected_total_minutes: {projected_total_seconds / 60:.2f}")
            if projected_total_seconds > 90 * 60:
                raise SystemExit(
                    f"Projected total training time exceeds 90 minutes ({projected_total_seconds / 60:.2f} minutes). Stopping now as requested."
                )

    wall_seconds = time.perf_counter() - start
    avg_loss = total_loss / max(1, total_items)
    return wall_seconds, avg_loss, batch_count


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        input_ids, attention_mask, labels = [tensor.to(DEVICE) for tensor in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=1)
        preds.extend(batch_preds.cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return targets, preds


def main() -> None:
    set_seed(42)
    ram_report()

    df = pd.read_csv(INPUT_PATH)
    print("\n===== Data =====")
    print(f"Loaded {INPUT_PATH.name} shape: {df.shape}")
    print(f"Loaded columns: {df.columns.tolist()}")
    required_cols = {"text", "label", "clean_text"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"Required columns missing. Expected at least {sorted(required_cols)}")

    sampled_df = sample_dataframe(df)
    print(f"Sampled rows: {len(sampled_df)}")
    print("Sample class balance:")
    print(sampled_df["label"].value_counts().sort_index().to_string())

    train_df, test_df = train_test_split(
        sampled_df,
        test_size=0.2,
        random_state=42,
        stratify=sampled_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if len(train_df) != TRAIN_ROWS or len(test_df) != TEST_ROWS:
        print(
            f"Warning: expected {TRAIN_ROWS}/{TEST_ROWS} split, got {len(train_df)}/{len(test_df)}."
        )

    print(f"Train/test sizes: {len(train_df)} / {len(test_df)}")
    print("Train class balance:")
    print(train_df["label"].value_counts().sort_index().to_string())
    print("Test class balance:")
    print(test_df["label"].value_counts().sort_index().to_string())

    train_texts = train_df["text"].fillna("").astype(str).tolist()
    test_texts = test_df["text"].fillna("").astype(str).tolist()
    train_labels = train_df["label"].astype(int).tolist()
    test_labels = test_df["label"].astype(int).tolist()

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    memory_guard("before_model_load")

    batch_size = INITIAL_BATCH_SIZE
    model = None
    optimizer = None
    train_loader = None
    test_loader = None
    while True:
        print("\n===== Tokenization =====")
        print(f"Tokenizer model: {MODEL_NAME}")
        print(f"max_length: {MAX_LENGTH}")
        print("truncation: True")
        print("padding: max_length")
        print(f"batch_size: {batch_size}")

        try:
            train_loader, test_loader = make_dataloaders(
                tokenizer,
                train_texts,
                train_labels,
                test_texts,
                test_labels,
                batch_size=batch_size,
            )
            memory_guard("after_tokenization")
            gc.collect()
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
            memory_guard("after_model_load")
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

            print("\n===== Architecture =====")
            print(f"device: {DEVICE}")
            print(f"model: {MODEL_NAME}")
            print("epochs: 1")
            print("optimizer: AdamW(lr=5e-5)")
            print("loss: CrossEntropyLoss")

            train_start = time.perf_counter()
            epoch_seconds, avg_loss, batches = train_one_epoch(model, train_loader, optimizer)
            train_wall_seconds = time.perf_counter() - train_start
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message and batch_size > 4:
                print(f"Memory error encountered at batch size {batch_size}. Retrying with batch size 4.")
                batch_size = 4
                model = None
                optimizer = None
                train_loader = None
                test_loader = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

    print("\n===== Training =====")
    print(f"epoch_batches: {batches}")
    print(f"epoch_loss: {avg_loss:.6f}")
    print(f"epoch_training_seconds: {epoch_seconds:.2f}")
    print(f"wall_clock_training_seconds: {train_wall_seconds:.2f}")

    y_true, y_pred = evaluate(model, test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    print("\n===== Test Evaluation =====")
    print(f"accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("\n===== Saved Artifacts =====")
    print(f"saved_model_dir: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"{path.name}\t{path.stat().st_size}")

    print("\n===== Summary =====")
    print("This Phase 5.2 run used only 2,000 rows sampled from delaked_data.csv (1,600 train / 400 test).")
    print("It is a small-scale proof-of-concept and is not directly comparable to the LSTM/TF-IDF results trained on larger samples.")


if __name__ == "__main__":
    main()
