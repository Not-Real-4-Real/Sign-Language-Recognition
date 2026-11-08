import json
import torch
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix

from models.gru import GRUClassifier
from src.data.data_loader import get_loaders


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "GRU_best.pth"
LABEL_PATH = "data/processed/id2label.json"

BATCH_SIZE = 32
NUM_CLASSES = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ============================================================
# LOAD MODEL
# ============================================================

model = GRUClassifier(
    input_size=150,
    hidden_size=128,
    num_layers=2,
    num_classes=NUM_CLASSES,
    dropout=0.2,
    bidirectional=False
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print(f"Loaded checkpoint: {CHECKPOINT_PATH}")


# ============================================================
# LOAD DATA
# ============================================================

train_loader, valid_loader, test_loader, train_eval_loader = get_loaders(
    batch_size=BATCH_SIZE
)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    id2label = json.load(f)

# JSON keys are usually strings, so convert them to integer indices.
label_names = [
    id2label[str(i)] for i in range(NUM_CLASSES)
]


# ============================================================
# EVALUATE BY PARTICIPANT
# ============================================================

def evaluate_by_participant(model, loader, device):

    dataset = loader.dataset
    results = []

    model.eval()

    participant_ids = sorted(
        dataset.df["participant_id"].unique()
    )

    for participant_id in participant_ids:

        indices = dataset.df.index[
            dataset.df["participant_id"] == participant_id
        ].tolist()

        subset = Subset(dataset, indices)

        participant_loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=loader.collate_fn
        )

        correct = 0
        total = 0

        with torch.no_grad():

            for batch in participant_loader:

                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
                lengths = batch["lengths"]

                outputs = model(features, lengths)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total

        results.append({
            "participant_id": participant_id,
            "samples": total,
            "accuracy": accuracy
        })

    return pd.DataFrame(results)


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

def collect_predictions(model, loader, device):

    all_labels = []
    all_predictions = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]

            outputs = model(features, lengths)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions


# ============================================================
# PARTICIPANT-LEVEL RESULTS
# ============================================================

valid_results = evaluate_by_participant(
    model, valid_loader, device
)

test_results = evaluate_by_participant(
    model, test_loader, device
)

print("\n========== VALIDATION PERFORMANCE ==========")
print(valid_results.to_string(index=False))

print("\n========== TEST PERFORMANCE ==========")
print(test_results.to_string(index=False))


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

y_true, y_pred = collect_predictions(
    model, test_loader, device
)

print("\n========== TEST CLASSIFICATION REPORT ==========")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=label_names,
    digits=4,
    zero_division=0
))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

print("\n========== TEST CONFUSION MATRIX ==========")
print(cm)


# ============================================================
# PARTICIPANT-CLASSIFICATION-LEVEL RESULTS
# ============================================================

def evaluate_participant_by_class(model, loader, device, label_names):

    dataset = loader.dataset
    records = []

    model.eval()

    participant_ids = sorted(
        dataset.df["participant_id"].unique()
    )

    with torch.no_grad():

        for participant_id in participant_ids:

            indices = dataset.df.index[
                dataset.df["participant_id"] == participant_id
            ].tolist()

            subset = Subset(dataset, indices)

            participant_loader = DataLoader(
                subset,
                batch_size=32,
                shuffle=False,
                collate_fn=loader.collate_fn
            )

            all_labels = []
            all_predictions = []

            for batch in participant_loader:

                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
                lengths = batch["lengths"]

                outputs = model(features, lengths)
                predictions = outputs.argmax(dim=1)

                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())

            all_labels = torch.tensor(all_labels)
            all_predictions = torch.tensor(all_predictions)

            for class_id, class_name in enumerate(label_names):

                class_mask = all_labels == class_id

                total = class_mask.sum().item()

                if total == 0:
                    accuracy = float("nan")
                else:
                    correct = (
                        all_predictions[class_mask] == class_id
                    ).sum().item()

                    accuracy = 100 * correct / total

                records.append({
                    "participant_id": participant_id,
                    "class": class_name,
                    "samples": total,
                    "accuracy": accuracy
                })

    return pd.DataFrame(records)


valid_class_results = evaluate_participant_by_class(
    model, valid_loader, device, label_names
)

test_class_results = evaluate_participant_by_class(
    model, test_loader, device, label_names
)

print("\nVALIDATION: PARTICIPANT x CLASS")
print(valid_class_results.pivot(
    index="class",
    columns="participant_id",
    values="accuracy"
).round(1))

print("\nTEST: PARTICIPANT x CLASS")
print(test_class_results.pivot(
    index="class",
    columns="participant_id",
    values="accuracy"
).round(1))
