import ast
import json
import re

import pandas as pd

import src.dataset_preparation.constants as const


def is_noise_category(category: str) -> bool:
    cat_lower: str = category.lower()
    return any(keyword in cat_lower for keyword in const.NOISE_KEYWORDS)


# Parse <think> section
def extract_list(answer):
    answer = answer.split("</think>")[-1].strip()
    try:
        match = re.search(r"\[.*?\]", answer, re.DOTALL)
        if match:
            return ast.literal_eval(match.group())
        else:
            return None
    except Exception:
        return None


def main() -> None:
    # LLM filtered noisy categories
    path = "../data/filtered_categories.csv"
    df = pd.read_csv(path)
    df["valid_idx"] = df["deepseek_answer"].apply(extract_list)

    graph = json.loads(
        open("../data/wiki_graph_dedup.json", "r").read()
    )  # read the graph

    for _, row in df.iterrows():
        title = row["title"]
        valid_idx = row["valid_idx"]  # valid categories that are meaningful

        # Remove noisy categories
        all_categories = graph["nodes"][title]["categories"]
        filtered_categories = [
            all_categories[i] for i in valid_idx if i < len(all_categories)
        ]
        graph["nodes"][title]["categories"] = filtered_categories

    titles_to_remove = []
    # If an article after category cleaning still contains noisy cats — drop it
    for title, node in graph["nodes"].items():
        categories = node.get("categories", [])

        if any(is_noise_category(cat) for cat in categories):
            titles_to_remove.append(title)

    for title in titles_to_remove:
        del graph["nodes"][title]

    print(
        f"[INFO] Removed {len(titles_to_remove)} noisy nodes. Remaining nodes: {len(graph['nodes'])}"
    )

    remaining_nodes = set(graph["nodes"].keys())
    # Remove unexistent edges
    graph["edges"] = [
        (src, dst)
        for src, dst in graph["edges"]
        if src in remaining_nodes and dst in remaining_nodes
    ]

    print(f"[INFO] Remaining edges after cleanup: {len(graph['edges'])}")

    # Save graph with filtered categories
    with open("../data/wiki_graph_filtered_v2.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
