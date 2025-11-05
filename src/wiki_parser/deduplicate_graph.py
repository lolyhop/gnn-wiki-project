import json
import typing as tp
from pathlib import Path


def load_graph(path: Path) -> tp.Dict[str, tp.Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_graph(graph: tp.Dict[str, tp.Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"Saved deduplicated graph to {path}")


def deduplicate_graph(
    graph: tp.Dict[str, tp.Any], make_undirected: bool = True
) -> tp.Dict[str, tp.Any]:
    nodes = graph["nodes"]
    edges = graph["edges"]

    print(f"Original: {len(nodes)} nodes, {len(edges)} edges")

    # Convert to set of tuples
    edge_set: tp.Set[tp.Tuple[str, str]] = set()

    for src, dst in edges:
        if src == dst:
            continue  # remove self-loop
        edge_set.add((src, dst))

    # Ensure (A,B) == (B,A)
    if make_undirected:
        undirected = set()
        for a, b in edge_set:
            key = tuple(sorted((a, b)))
            undirected.add(key)
        edge_set = undirected

    # Convert back to list
    dedup_edges = list(edge_set)

    print(f"Deduplicated edges: {len(dedup_edges)}")

    # Reattach nodes
    return {
        "nodes": nodes,
        "edges": dedup_edges,
    }


def main() -> None:
    INPUT = Path("./data/wiki_graph.json")
    OUTPUT = Path("./data/wiki_graph_dedup.json")

    graph = load_graph(INPUT)
    graph = deduplicate_graph(graph, make_undirected=True)
    save_graph(graph, OUTPUT)


if __name__ == "__main__":
    main()
