import torch
from torch_geometric.data import Data


def read_and_split(
    path: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> tuple[Data, Data, Data]:

    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"

    print(f"Loading full graph from {path}...")
    loaded_content = torch.load(path, weights_only=False)
    data = loaded_content

    num_nodes = data.num_nodes
    print(f"Graph loaded: {num_nodes} nodes, {data.num_edges} edges")

    # 1. Nodes deterministic permutation
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)

    # 2. Calculate splits boundaries
    n_train = int(num_nodes * ratios[0])
    n_val = int(num_nodes * ratios[1])

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    # 3. Create train/val/test masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    # Stats
    print("-" * 30)
    print(f"Transductive Split Created:")
    print(f"Train Mask: {data.train_mask.sum().item()} nodes")
    print(f"Val Mask:   {data.val_mask.sum().item()} nodes")
    print(f"Test Mask:  {data.test_mask.sum().item()} nodes")
    print(f"Edges kept: {data.num_edges} (100%)")
    print("-" * 30)

    return data
