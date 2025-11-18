import json
import typing as tp
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

import src.dataset_preparation.constants as const

if torch.cuda.is_available():
    device = "cuda"
    print("[INFO] Using CUDA GPU")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    print("[INFO] Using Apple MPS GPU")
else:
    device = "cpu"
    print("[INFO] Using CPU")


def load_graph(path: Path) -> tp.Dict[str, tp.Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_raw_label(node: tp.Dict[str, tp.Any]) -> str:
    """Concatenate and clean Wikipedia categories for a node"""
    categories = node.get("categories", [])
    clean_cats = [
        cat.replace("Category:", "").strip() for cat in categories if cat.strip()
    ]
    return ", ".join(clean_cats)


def build_prompts(
    titles: tp.List[str], nodes: tp.Dict[str, tp.Any], topics: tp.List[str]
) -> tp.List[str]:
    prompts = []
    for title in titles:
        node_cats = build_raw_label(nodes[title])
        prompt = f'Article categories: "{node_cats}"'
        prompts.append(prompt)
    return prompts


def main() -> None:
    INPUT_GRAPH = Path("./data/wiki_graph_filtered_v2.json")
    OUTPUT_LABELS = Path("./data/wiki_node_labels.json")

    print(f"[INFO] Path to graph: {INPUT_GRAPH}")
    graph = load_graph(INPUT_GRAPH)
    nodes = graph["nodes"]
    titles: tp.List[str] = list(nodes.keys())
    topic_names = list(const.IT_TOPICS.keys())
    topic_texts = list(const.IT_TOPICS.values())

    # Prepare node prompts
    print("[INFO] Building prompts for query embeddings...")
    query_prompts = build_prompts(titles, nodes, topic_names)

    # Load embedding model
    model_name = "intfloat/multilingual-e5-large-instruct"
    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Embed nodes (queries)
    print("[INFO] Generating nodes embeddings...")
    query_embeddings = model.encode(
        query_prompts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Embed categories (labels)
    print("[INFO] Generating class embeddings...")
    doc_embeddings = model.encode(
        topic_texts, convert_to_numpy=True, normalize_embeddings=True
    )

    # Compute cosine similarity and assign a class
    print("[INFO] Calculating cosine similarity...")
    similarities = util.cos_sim(
        query_embeddings, doc_embeddings
    )  # shape [n_nodes, n_topics]

    # Threshold for assigning "Other" class
    THRESHOLD = 0.4

    print("[INFO] Assigning classes...")
    node_labels: tp.Dict[str, tp.Dict[str, tp.Any]] = {}
    for i, title in enumerate(titles):
        sims = similarities[i]
        max_idx = int(torch.argmax(sims).item())
        max_score = float(sims[max_idx].item())
        if max_score < THRESHOLD:
            assigned_class = "Other"
        else:
            assigned_class = topic_names[max_idx]

        node_labels[title] = {
            "assigned_class": assigned_class,
            "score": max_score,
            "categories": nodes[title].get("categories", []),
        }

    # Save results
    print(f"[INFO] Saving node labels -> {OUTPUT_LABELS}")
    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f:
        json.dump(node_labels, f, ensure_ascii=False)

    print(f"[INFO] Saved labels for {len(node_labels)} nodes.")


if __name__ == "__main__":
    main()
