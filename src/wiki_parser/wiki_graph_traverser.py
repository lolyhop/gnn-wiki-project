import json
import random
import typing as tp
import time
from pathlib import Path

from tqdm import tqdm

from src.wiki_parser.wiki_api import WikipediaAPI
import src.wiki_parser.constants as const


class WikiGraphTraverser:
    def __init__(
        self,
        wiki_api: WikipediaAPI,
        max_depth: int = 2,
        max_nodes: int = 1000,
        links_per_page: int = 20,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        autosave_every: int = 200,
        save_path: tp.Optional[Path] = None,
    ):
        self.wiki = wiki_api
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.links_per_page = links_per_page
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.autosave_every = autosave_every
        self.save_path = save_path

        self.visited: tp.Set[str] = set()
        self.graph: tp.Dict[str, tp.Any] = {"nodes": {}, "edges": []}

        if self.save_path and Path(self.save_path).exists():
            self._resume_from_existing()

    def _resume_from_existing(self) -> None:
        """Load partially built graph if available."""
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            self.graph = existing
            self.visited = set(existing["nodes"].keys())
            print(
                f"[RESUME] Loaded {len(self.graph['nodes'])} nodes and "
                f"{len(self.graph['edges'])} edges from {self.save_path}"
            )
        except Exception as e:
            print(f"[WARN] Could not resume from {self.save_path}: {e}")

    def _fetch_with_retry(self, title: str) -> tp.Optional[const.Article]:
        """Try fetching a page with retries."""
        for attempt in range(self.retry_attempts):
            try:
                page = self.wiki.get_page(title)
                if page:
                    return page
            except Exception as e:
                print(f"[WARN] Failed to fetch '{title}' (attempt {attempt+1}): {e}")
            time.sleep(self.retry_delay)
        return None

    def _autosave(self) -> None:
        """Save current progress to disk."""
        if not self.save_path:
            return
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)
        print(
            f"[AUTO-SAVE] Saved progress: {len(self.graph['nodes'])} nodes, "
            f"{len(self.graph['edges'])} edges → {self.save_path}"
        )

    def build_graph(
        self, seed_articles: tp.List[const.Article]
    ) -> tp.Dict[str, tp.Any]:
        """Traverse Wikipedia starting from given seed articles."""
        if len(self.graph["nodes"]) > 0:
            all_titles = set(self.graph["nodes"].keys())
            # frontier = nodes that have no outgoing edges in saved data (i.e., probably last batch)
            frontier = [
                t
                for t in all_titles
                if all(
                    l not in all_titles for l in self.graph["nodes"][t].get("links", [])
                )
            ]
            print(f"Unmarked: {len(frontier)}")

            for f in frontier:
                self.visited.discard(f)
            print(f"Resuming traversal from {len(frontier)} frontier nodes...")
            queue = [(f, 0) for f in frontier]
        else:
            queue = [(a["title"], 0) for a in seed_articles]

        pbar = tqdm(total=self.max_nodes, desc="Building Wiki Graph", unit="node")

        while queue and len(self.graph["nodes"]) < self.max_nodes:
            title, depth = queue.pop(0)
            if depth > self.max_depth or title in self.visited:
                continue

            self.visited.add(title)
            page = self._fetch_with_retry(title)
            if not page:
                continue

            self.graph["nodes"][title] = page
            pbar.update(1)

            links = page.get("links", [])
            num_links = random.randint(1, self.links_per_page)
            selected_links = random.sample(links, min(len(links), num_links))

            for link in selected_links:
                self.graph["edges"].append((title, link))
                if (
                    link not in self.visited
                    and len(self.graph["nodes"]) < self.max_nodes
                ):
                    queue.append((link, depth + 1))

            # autosave every N nodes
            if len(self.graph["nodes"]) % self.autosave_every == 0:
                self._autosave()

        pbar.close()

        unique_edges = set()
        for a, b in self.graph["edges"]:
            # (A,B) == (B,A)
            edge = tuple(sorted((a, b)))
            unique_edges.add(edge)
        self.graph["edges"] = list(unique_edges)

        return self.graph

    def save(self, path: tp.Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)
        print(
            f"Saved graph with {len(self.graph['nodes'])} nodes "
            f"and {len(self.graph['edges'])} edges → {path}"
        )


def main() -> None:
    wiki = WikipediaAPI(language=const.LANGUAGE, user_agent=const.USER_AGENT)
    output_path = Path("./data/wiki_graph.json")

    traverser = WikiGraphTraverser(
        wiki_api=wiki,
        max_depth=const.MAX_DEPTH_PER_GRAPH,
        max_nodes=const.MAX_NODES_PER_GRAPH,
        links_per_page=const.MAX_LINKS_PER_PAGE,
        retry_attempts=const.NUM_OF_RETRIES,
        retry_delay=const.RETRY_DELAY,
        autosave_every=const.SAVE_STEPS,
        save_path=output_path,
    )

    with open("./data/seed_articles.json", "r", encoding="utf-8") as f:
        seeds = json.load(f)

    graph = traverser.build_graph(seeds)
    traverser.save(output_path)


if __name__ == "__main__":
    main()
