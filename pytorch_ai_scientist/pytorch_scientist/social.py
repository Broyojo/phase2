"""X (Twitter) search helpers using xdk.

Provides lightweight wrappers to fetch threads from selected authors and
prepare them for downstream LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from requests.exceptions import HTTPError

from pytorch_scientist.config import XSearchConfig
from pytorch_scientist.utils.logging import get_logger

logger = get_logger("social")


# ---------- Models ----------


@dataclass
class Post:
    """Minimal tweet representation with helpers."""

    id: str
    text: str
    created_at: Optional[str] = None
    author_id: Optional[str] = None
    conversation_id: Optional[str] = None
    note_tweet: Optional[Dict[str, Any]] = None
    referenced_tweets: Optional[List[Dict[str, Any]]] = None

    # Expanded referenced tweet when available (e.g., for RTs)
    expanded_referenced_tweet: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def get_full_text(self) -> str:
        """Return full text, preferring original content for RTs and note tweets."""
        # If this is an RT and we have the expanded original tweet
        if self.is_retweet() and self.expanded_referenced_tweet:
            original_text = self.expanded_referenced_tweet.get("text", "")
            # Check if original has note_tweet
            if "note_tweet" in self.expanded_referenced_tweet:
                note_text = self.expanded_referenced_tweet["note_tweet"].get("text")
                if note_text:
                    original_text = note_text
            return f"RT: {original_text}"

        # Otherwise use note_tweet if available for long tweets
        if self.note_tweet and "text" in self.note_tweet:
            return self.note_tweet["text"]

        return self.text or ""

    def is_retweet(self) -> bool:
        """Check if this post is a retweet."""
        if not self.referenced_tweets:
            return False
        return any(ref.get("type") == "retweeted" for ref in self.referenced_tweets)


@dataclass
class ThreadSummary:
    """Full-text thread summary for a conversation."""

    id: str  # root tweet id (earliest in convo for this author)
    author_id: str
    created_at: Optional[str]
    text: str  # concatenated thread text

    def to_summary_string(self) -> str:
        """Render a string suitable for LLM context."""
        return (
            f"Author: {self.author_id}\n"
            f"Created: {self.created_at}\n"
            f"Thread Text:\n{self.text}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "created_at": self.created_at,
            "text": self.text,
        }


# ---------- X client / fields ----------


TWEET_FIELDS = [
    "id",
    "text",
    "author_id",
    "created_at",
    "conversation_id",
    "referenced_tweets",
    "entities",
    "note_tweet",
]

EXPANSIONS = [
    "referenced_tweets.id",
    "referenced_tweets.id.author_id",
]


def _get_client(x_config: XSearchConfig):
    """Lazily create the xdk client or a mock if unavailable."""
    try:
        from xdk import Client

        if not x_config.api_key:
            raise ValueError("X_API_KEY not set")

        return Client(bearer_token=x_config.api_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Falling back to MockXClient: {exc}")
        return MockXClient()


# ---------- Core helpers ----------


def process_page_with_includes(page) -> List[Post]:
    """Process a page of results, expanding referenced tweets from includes."""
    raw_data = getattr(page, "data", None) or []
    includes = getattr(page, "includes", None) or {}

    # Build a lookup of referenced tweets
    referenced_tweets_map = {}
    if "tweets" in includes:
        for tweet in includes["tweets"]:
            referenced_tweets_map[tweet.get("id")] = tweet

    posts: List[Post] = []
    for item in raw_data:
        post = Post(**item)

        # If this post references other tweets, try to expand them
        if post.referenced_tweets:
            for ref in post.referenced_tweets:
                ref_id = ref.get("id")
                if ref_id and ref_id in referenced_tweets_map:
                    post.expanded_referenced_tweet = referenced_tweets_map[ref_id]
                    break  # Just use the first one (for RTs, there's typically only one)

        posts.append(post)

    return posts


def _build_query(authors: List[str], query: str, include_retweets: bool) -> str:
    from_clause = " OR ".join(f"from:{u}" for u in authors) if authors else ""
    rt_filter = "" if include_retweets else "-is:retweet"

    if from_clause and query:
        full_query = f"({from_clause}) ({query}) {rt_filter}".strip()
    elif from_clause:
        full_query = f"{from_clause} {rt_filter}".strip()
    else:
        full_query = f"{query} {rt_filter}".strip()
    return full_query


def search_author_posts(
    authors: List[str],
    query: str,
    x_config: XSearchConfig,
    client=None,
) -> List[Post]:
    """Search tweets from given authors matching `query`.

    Returns a list of Post objects (processes pages up to max_results).
    """

    client = client or _get_client(x_config)
    full_query = _build_query(authors, query, x_config.include_retweets)
    posts: List[Post] = []

    search_fn_name = "search_all" if x_config.use_full_archive else "search_recent"
    search_fn = getattr(client.posts, search_fn_name)

    try:
        for page in search_fn(
            query=full_query,
            max_results=min(x_config.max_results, 100),
            tweet_fields=TWEET_FIELDS,
            expansions=EXPANSIONS,
        ):
            posts.extend(process_page_with_includes(page))

            if len(posts) >= x_config.max_results:
                break

    except HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (429, 503):
            # rate-limited or temporary issue -> return whatever we got
            return posts
        raise

    return posts


def fetch_author_conversation(
    author_id: str,
    conv_id: str,
    x_config: XSearchConfig,
    client=None,
) -> List[Post]:
    """Fetch recent tweets by `author_id` in conversation `conv_id`."""
    if not author_id:
        return []

    client = client or _get_client(x_config)
    query = f"conversation_id:{conv_id} from:{author_id} -is:retweet"
    posts: List[Post] = []

    try:
        for page in client.posts.search_recent(
            query=query,
            max_results=100,
            tweet_fields=TWEET_FIELDS,
            expansions=EXPANSIONS,
        ):
            posts.extend(process_page_with_includes(page))

    except HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (429, 503):
            return []
        raise

    return posts


def build_thread_simple(
    root: Post,
    x_config: XSearchConfig,
    client=None,
) -> ThreadSummary:
    """Build a thread by fetching author conversation and concatenating text."""
    author_id = root.author_id or ""
    conv_id = root.conversation_id or root.id

    posts = fetch_author_conversation(author_id, conv_id, x_config, client=client)

    # Ensure root is present in case search_recent missed it
    if not any(p.id == root.id for p in posts):
        posts.append(root)

    if not posts:
        return ThreadSummary(
            id=root.id,
            author_id=author_id,
            created_at=root.created_at,
            text=root.get_full_text(),
        )

    # Oldest → newest
    posts.sort(key=lambda p: p.created_at or "")

    full_text = "\n\n".join(p.get_full_text() for p in posts if p.get_full_text())
    first = posts[0]

    return ThreadSummary(
        id=first.id,
        author_id=author_id,
        created_at=first.created_at,
        text=full_text,
    )


def get_threads_for_authors(
    authors: List[str],
    query: str,
    x_config: XSearchConfig,
    client=None,
) -> List[ThreadSummary]:
    """High-level entry: search, group by conversation, and return threads."""
    posts = search_author_posts(authors, query, x_config, client=client)

    # pick earliest tweet per conversation as root
    conv_to_root: Dict[str, Post] = {}
    for p in posts:
        conv_id = p.conversation_id or p.id
        current = conv_to_root.get(conv_id)
        if current is None or (p.created_at or "") < (current.created_at or ""):
            conv_to_root[conv_id] = p

    # sort roots by time for nicer ordering
    roots = sorted(
        conv_to_root.values(),
        key=lambda p: p.created_at or "",
        reverse=False,
    )

    threads: List[ThreadSummary] = []
    for root in roots:
        threads.append(build_thread_simple(root, x_config, client=client))

    return threads


# ---------- Mock client for tests / missing API key ----------


class MockPosts:
    def __init__(self, tweets: List[Dict[str, Any]]):
        self._tweets = tweets

    def _paginate(self, query: str, max_results: int, **_: Any):  # noqa: ANN401
        class Page:
            def __init__(self, data: List[Dict[str, Any]]):
                self.data = data
                self.includes = {"tweets": []}

        yield Page(self._tweets[:max_results])

    def search_recent(self, **kwargs):
        return self._paginate(**kwargs)

    def search_all(self, **kwargs):
        return self._paginate(**kwargs)


class MockXClient:
    """Simplified mock xdk client."""

    def __init__(self):
        sample = [
            {
                "id": "1",
                "text": "Mock tweet about attention",
                "author_id": "author1",
                "created_at": "2024-01-01T00:00:00Z",
                "conversation_id": "c1",
                "referenced_tweets": None,
            },
            {
                "id": "2",
                "text": "Another mock tweet",
                "author_id": "author2",
                "created_at": "2024-01-02T00:00:00Z",
                "conversation_id": "c2",
                "referenced_tweets": None,
            },
        ]
        self.posts = MockPosts(sample)
