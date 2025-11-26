import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def read_and_split_inductive(
    path: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> tuple[Data, Data, Data]:

    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"

    print(f"Loading data from {path}...")
    loaded_content = torch.load(path, weights_only=False)
    data = loaded_content

    num_nodes = data.num_nodes
    print(f"Original graph: {num_nodes} nodes, {data.num_edges} edges")

    # 1. Shuffle nodes deterministically
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)

    # 2. Calculate split indices
    n_train = int(num_nodes * ratios[0])
    n_val = int(num_nodes * ratios[1])

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    print("Splitting graph into induced subgraphs...")

    # 3. Create induced subgraphs

    # Train
    edge_index_train, _ = subgraph(train_idx, data.edge_index, relabel_nodes=True)
    train_data = Data(
        x=data.x[train_idx], edge_index=edge_index_train, y=data.y[train_idx]
    )

    # Val
    edge_index_val, _ = subgraph(val_idx, data.edge_index, relabel_nodes=True)
    val_data = Data(x=data.x[val_idx], edge_index=edge_index_val, y=data.y[val_idx])

    # Test
    edge_index_test, _ = subgraph(test_idx, data.edge_index, relabel_nodes=True)
    test_data = Data(x=data.x[test_idx], edge_index=edge_index_test, y=data.y[test_idx])

    # Stats
    print("-" * 30)
    print(f"Train Graph: {train_data.num_nodes} nodes, {train_data.num_edges} edges")
    print(f"Val Graph:   {val_data.num_nodes} nodes, {val_data.num_edges} edges")
    print(f"Test Graph:  {test_data.num_nodes} nodes, {test_data.num_edges} edges")
    print("-" * 30)

    return train_data, val_data, test_data
