import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from src.model_training.utils import read_and_split


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        # Layer 1: Multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.25)
        # Layer 2: Output layer (concat=False to merge heads)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=heads, concat=False, dropout=0.25
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        print(x.shape)
        return x


def calculate_metrics(y_true, y_pred):
    """Calculates weighted metrics."""
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return acc, p, r, f1


def save_plots(history, filename="gat_metrics.png"):
    """Plots training history."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    # Metrics Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Acc", color="green")
    plt.plot(epochs, history["val_f1"], label="Val F1", color="orange", linestyle="--")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_full_batch(model, data, mask):
    """ Helper for full-graph evaluation using masks. """
    model.eval()
    with torch.no_grad():
        # Forward pass on the whole graph
        out = model(data.x, data.edge_index)
        
        # Filter by mask
        pred = out[mask].argmax(dim=1)
        y_true = data.y[mask]
        
        acc, _, _, f1 = calculate_metrics(y_true, pred)
    return acc, f1, y_true, pred


def run_gat_training():
    # --- CONFIG ---
    PATH = "./data/wiki_it_graph_scibert_feats.pt"
    PLOT_PATH = "./plots"
    os.makedirs(PLOT_PATH, exist_ok=True)

    HIDDEN_DIM = 32  # Total hidden size will be 64 * 8 = 512
    HEADS = 8
    LR = 3e-4
    EPOCHS = 100
    BATCH_SIZE = 16 # Number of target nodes per batch
    
    # Neighbor Sampling config
    # [10, 10] means: sample 10 neighbors for hop-1, and 10 for hop-2
    NUM_NEIGHBORS = [10, 10] 

    # Mac MPS check (GAT often unstable on MPS, prefer CPU if issues arise)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        DEVICE = torch.device("cpu") 
    
    print(f"Using device: {DEVICE}")

    # --- DATA LOADING ---
    # 1. Load single Transductive graph
    data = read_and_split(PATH)
    data = data.to(DEVICE) # Move full graph to device for validation
    
    # 2. Setup NeighborLoader for Training
    # This enables mini-batch training on large graphs
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.train_mask, # Only sample batches from Train nodes
        shuffle=True
    )

    num_features = data.x.shape[1]
    num_classes = len(torch.unique(data.y))
    print(f"Features: {num_features}, Classes: {num_classes}")
    print(f"Train Batches: {len(train_loader)}")

    # --- MODEL SETUP ---
    model = GAT(num_features, HIDDEN_DIM, num_classes, heads=HEADS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0

    print("\nStarting Batched Transductive Training...")

    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN (Mini-Batch) ---
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass on sampled subgraph
            out = model(batch.x, batch.edge_index)
            
            # Slice output: NeighborLoader places target nodes first
            # We only compute loss on the target nodes (batch_size)
            target_out = out[:batch.batch_size]
            target_y = batch.y[:batch.batch_size]
            
            print(target_out.shape, target_y.shape)
            loss = criterion(target_out, target_y)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)

        # --- VALIDATION (Full-Batch) ---
        # For small/medium graphs, full-batch eval is standard and accurate
        acc_val, f1_val, _, _ = evaluate_full_batch(model, data, data.val_mask)

        # Logging
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc_val)
        history["val_f1"].append(f1_val)

        if f1_val > best_val_f1:
            best_val_f1 = f1_val

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: Loss: {avg_loss:.4f} | Val Acc: {acc_val:.4f} | Val F1: {f1_val:.4f}"
            )

    print(f"\nTraining finished. Best Val F1: {best_val_f1:.4f}")
    save_plots(history, filename=f"{PLOT_PATH}/gat_metrics.png")

    # --- TEST EVALUATION ---
    print("\nEvaluating on Test Set...")
    _, _, y_true, y_pred = evaluate_full_batch(model, data, data.test_mask)
    
    # Classification Report expects CPU numpy arrays
    print(classification_report(y_true.cpu(), y_pred.cpu(), digits=4))


if __name__ == "__main__":
    run_gat_training()
