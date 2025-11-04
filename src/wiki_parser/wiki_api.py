import random
import typing as tp
import wikipediaapi

import src.wiki_parser.constants as const


class WikipediaAPI:
    """Wikipedia API wrapper."""

    def __init__(
        self, language: str = const.LANGUAGE, user_agent: str = const.USER_AGENT
    ):
        self.api = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)

    def get_page(self, title: str) -> tp.Optional[const.Article]:
        """Fetch a Wikipedia page and return structured data."""
        page = self.api.page(title)
        if not page.exists():
            return None

        # Filter out non-article links
        links: tp.List[str] = [link for link in page.links.keys() if ":" not in link]

        return {
            "title": page.title,
            "categories": list(page.categories.keys()),
            "link": page.fullurl,
            "text": page.text,
            "links": links,
        }

    def get_category_members(
        self,
        category_name: str,
        limit: tp.Optional[int] = None,
        shuffle: bool = False,
    ) -> tp.List[str]:
        """Return article titles under a category."""
        category = self.api.page(category_name)
        members = [p.title for p in category.categorymembers.values() if p.ns == 0]
        if shuffle:
            random.shuffle(members)
        return members[:limit] if limit else members

    def exists(self, title: str) -> bool:
        """Check if a page exists."""
        return self.api.page(title).exists()
