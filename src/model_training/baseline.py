import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from src.model_training.utils import read_and_split


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x


def calculate_metrics(y_true, y_pred):
    """Calculates accuracy, precision, recall, and f1 (weighted)."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return acc, p, r, f1


def run_evaluation(model, loader, device):
    """Runs inference on a loader and checks metrics."""
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu()

            preds_list.append(pred)
            targets_list.append(y)

    # Concatenate all batches
    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(targets_list).numpy()

    # Calculate metrics
    loss = 0.0
    acc, p, r, f1 = calculate_metrics(y_true, y_pred)

    return loss, acc, f1, y_true, y_pred


def save_plots(history, filename="baseline_metrics.png"):
    """Saves loss and accuracy plots."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    # plt.plot(epochs, history["val_loss"], label="Val Loss") # Optional if we calculate val loss
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot 2: Accuracy & F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Acc", color="green")
    plt.plot(epochs, history["val_f1"], label="Val F1", color="orange", linestyle="--")
    plt.title("Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plots saved to {filename}")
    plt.close()


def run_baseline_training():
    # Config
    PATH = "./data/wiki_it_graph_scibert_feats.pt"
    PLOT_PATH = "plots"
    os.makedirs(PLOT_PATH, exist_ok=True)

    HIDDEN_DIM = 256
    LR = 2e-3
    EPOCHS = 100
    BATCH_SIZE = 64

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")

    print(f"Using device: {DEVICE}")

    # Load data
    data = read_and_split(PATH, seed=42)

    # Create TensorDatasets
    train_dataset = TensorDataset(data.x[data.train_mask], data.y[data.train_mask])
    val_dataset = TensorDataset(data.x[data.val_mask], data.y[data.val_mask])
    test_dataset = TensorDataset(data.x[data.test_mask], data.y[data.test_mask])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_features = data.x.shape[1]
    num_classes = len(torch.unique(data.y))

    print(f"Input Dim: {num_features}, Classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")

    # Init Model
    model = MLP(num_features, HIDDEN_DIM, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nStarting training...")
    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training loop
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Avg loss for the epoch
        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on full validation set
        _, acc_val, f1_val, _, _ = run_evaluation(model, val_loader, DEVICE)

        # Log metrics
        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(acc_val)
        history["val_f1"].append(f1_val)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {acc_val:.4f} | "
                f"Val F1: {f1_val:.4f}"
            )

        if f1_val > best_val_f1:
            best_val_f1 = f1_val

    print(f"\nTraining finished. Best Val F1: {best_val_f1:.4f}")

    # Plot metrics
    save_plots(history, filename=f"{PLOT_PATH}/baseline_metrics.png")

    # Test set evaluation
    print("\nEvaluating on Test Set...")
    _, _, _, y_true, y_pred = run_evaluation(model, test_loader, DEVICE)

    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    run_baseline_training()
