import json
import random
import typing as tp

import wikipediaapi

import src.wiki_parser.constants as const


def collect_articles() -> tp.List[const.Article]:
    wiki = wikipediaapi.Wikipedia(language=const.LANGUAGE, user_agent=const.USER_AGENT)
    pages: tp.Set[str] = set()

    # For each seed category save `ARTICLES_PER_SEED` article names
    for _category in const.SEED_CATEGORIES:
        category = wiki.page(_category)
        members = [p for p in category.categorymembers.values() if p.ns == 0]
        random.shuffle(members)
        pages.update(p.title for p in members[: const.ARTICLES_PER_SEED])

    # Collect articles data
    articles: tp.List[const.Article] = []
    for title in pages:
        page = wiki.page(title)
        if not page.exists():
            continue

        article: const.Article = {
            "title": title,
            "categories": list(page.categories.keys()),
            "link": page.fullurl,
            "text": page.text,
        }
        articles.append(article)

    return articles


def main() -> None:
    OUTPUT_PATH = "./data/seed_articles.json"
    data = collect_articles()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
