import time
import torch
import torch.nn as nn
import torch.optim as optim

from models.gru import GRUClassifier
from src.data.data_loader import get_loaders


# ---- 1. Data ----

train_loader, valid_loader, test_loader, train_eval_loader = get_loaders()


# ---- 2. Model ----

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = GRUClassifier(
    input_size=150,
    hidden_size=192,
    num_layers=2,
    num_classes=15,
    dropout=0.2,
    bidirectional=False
).to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)

print(f"Total parameters: {total_params:,}")


# ---- 3. Loss & optimizer ----

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-2
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=25,
    eta_min=1e-5
)


# ---- 4. Best model tracking ----

best_valid_loss = float("inf")
best_epoch = 0

best_model_path = "GRU_best.pth"


# ---- 5. Training loop ----

epochs = 25

start_time = time.perf_counter()

for epoch in range(epochs):

    epoch_start = time.perf_counter()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:

        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]

        optimizer.zero_grad()

        outputs = model(features, lengths)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        train_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += batch_size

    avg_train_loss = train_loss / train_total

    train_accuracy = (
        100 * train_correct / train_total
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():

        for batch in valid_loader:

            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]

            outputs = model(features, lengths)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            valid_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)

            valid_correct += (
                preds == labels
            ).sum().item()

            valid_total += batch_size

    avg_valid_loss = valid_loss / valid_total

    valid_accuracy = (
        100 * valid_correct / valid_total
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if avg_valid_loss < best_valid_loss:

        best_valid_loss = avg_valid_loss
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"  -> New best model saved "
            f"(Valid Loss: {best_valid_loss:.4f})"
        )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = time.perf_counter() - epoch_start


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Valid Loss: {avg_valid_loss:.4f} | "
        f"Valid Acc: {valid_accuracy:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_time:.2f}s"
    )


total_time = time.perf_counter() - start_time

print(f"\nTotal training time: {total_time:.2f}s")

print(
    f"Best model: Epoch {best_epoch} "
    f"| Valid Loss: {best_valid_loss:.4f}"
)


# ---- 6. Load best model ----

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device,
        weights_only=True
    )
)

print(f"Loaded best model from epoch {best_epoch}")


# ---- 7. Final Training Evaluation (Dropout OFF) ----

model.eval()

train_eval_correct = 0
train_eval_total = 0

with torch.no_grad():

    for batch in train_eval_loader:

        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]

        outputs = model(features, lengths)

        preds = outputs.argmax(dim=1)

        train_eval_correct += (
            preds == labels
        ).sum().item()

        train_eval_total += labels.size(0)

train_eval_accuracy = (
    100 * train_eval_correct / train_eval_total
)

print(
    f"Training accuracy with Dropout OFF: "
    f"{train_eval_accuracy:.2f}%"
)


# ---- 8. Final Test Evaluation ----

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for batch in test_loader:

        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]

        outputs = model(features, lengths)

        preds = outputs.argmax(dim=1)

        test_correct += (
            preds == labels
        ).sum().item()

        test_total += labels.size(0)

test_accuracy = 100 * test_correct / test_total

print(
    f"Test accuracy: {test_accuracy:.2f}%"
)