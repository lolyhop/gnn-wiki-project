import json
import random
import typing as tp

import src.wiki_parser.constants as const
from src.wiki_parser.wiki_api import WikipediaAPI


def collect_articles() -> tp.List[const.Article]:
    wiki = WikipediaAPI(language=const.LANGUAGE, user_agent=const.USER_AGENT)
    pages: tp.Set[str] = set()

    # For each seed category save `ARTICLES_PER_SEED` article names
    for _category in const.SEED_CATEGORIES:
        members = wiki.get_category_members(
            category_name=_category,
            limit=const.ARTICLES_PER_SEED,
            shuffle=True,
        )
        pages.update(members)

    # Collect articles data
    articles: tp.List[const.Article] = []
    for title in pages:
        page = wiki.get_page(title)
        if not page:
            continue
        articles.append(page)

    return articles


def main() -> None:
    OUTPUT_PATH = "./data/seed_articles.json"
    data = collect_articles()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
