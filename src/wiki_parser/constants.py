import typing as tp

# Constants
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
Article = tp.Dict[str, tp.Any]
