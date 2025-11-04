import typing as tp

# Traverser configuration
MAX_DEPTH_PER_GRAPH = 20
MAX_NODES_PER_GRAPH = 20000
MAX_LINKS_PER_PAGE = 5
NUM_OF_RETRIES = 3
RETRY_DELAY = 2.0
SAVE_STEPS = 30

# Parsers configuration
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15 "
    "(WikipediaGraphResearch/1.0; contact: egor@example.com)"
)

LANGUAGE = "en"

SEED_CATEGORIES = [
    "Category:Information technology",
    "Category:Computer science",
    "Category:Software engineering",
    "Category:Computer security",
    "Category:Artificial intelligence",
    "Category:Data management",
    "Category:Networking",
]
ARTICLES_PER_SEED = 30

# Data types
Article = tp.TypedDict(
    "Article",
    {
        "title": str,
        "categories": tp.List[str],
        "link": str,
        "text": str,
        "links": tp.List[str],
    },
)
