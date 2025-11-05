import json
import typing as tp
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import src.dataset_preparation.constants as const


def load_graph(path: Path) -> tp.Dict[str, tp.Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_raw_label(node: tp.Dict[str, tp.Any]) -> str:
    categories = node.get("categories", [])
    cat_str = " ".join(categories)
    return cat_str.strip()


def main() -> None:
    INPUT_GRAPH = Path("./data/wiki_graph_dedup.json")
    OUTPUT_VECTORIZER = Path(".data/tfidf_weights.json")
    OUTPUT_EMBEDDINGS = Path(".data/wiki_dedup_embds.json")
    print(f"[LOAD] Graph: {INPUT_GRAPH}")
    graph = load_graph(INPUT_GRAPH)

    nodes = graph["nodes"]
    titles: tp.List[str] = list(nodes.keys())

    print("[STEP] Building raw labels...")
    raw_labels: tp.List[str] = [build_raw_label(nodes[title]) for title in tqdm(titles)]

    print("[STEP] Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=const.TFIDF_MAX_FEATURES,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=const.TFIDF_NGRAM_RANGE,
    )

    X = vectorizer.fit_transform(raw_labels)
    print(f"[OK] TF-IDF matrix shape: {X.shape}")

    print(f"[SAVE] Saving vectorizer -> {OUTPUT_VECTORIZER}")
    joblib.dump(vectorizer, OUTPUT_VECTORIZER)

    print("[STEP] Converting embeddings...")
    embeddings_dict: tp.Dict[str, tp.List[float]] = {}

    for i, title in enumerate(titles):
        vec = X[i].toarray()[0].tolist()
        embeddings_dict[title] = vec

    print(f"[SAVE] Saving embeddings -> {OUTPUT_EMBEDDINGS}")
    with open(OUTPUT_EMBEDDINGS, "w", encoding="utf-8") as f:
        json.dump(embeddings_dict, f, ensure_ascii=False)

    print(f"[DONE] Saved {len(embeddings_dict)} embeddings.")


if __name__ == "__main__":
    main()
