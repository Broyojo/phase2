import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from requests.exceptions import HTTPError
from xdk import Client

# ---------- Models ----------


class Post(BaseModel):
    id: str
    text: str

    created_at: Optional[str] = None
    author_id: Optional[str] = None
    conversation_id: Optional[str] = None
    note_tweet: Optional[Dict[str, Any]] = None

    referenced_tweets: Optional[List[Dict[str, Any]]] = None

    # Store expanded referenced tweet data
    expanded_referenced_tweet: Optional[Dict[str, Any]] = None

    def get_full_text(self) -> str:
        """Get full text, handling RTs, note_tweets, and regular tweets"""
        # If this is a RT and we have the expanded original tweet
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
        """Check if this post is a retweet"""
        if not self.referenced_tweets:
            return False
        return any(ref.get("type") == "retweeted" for ref in self.referenced_tweets)


class ThreadSummary(BaseModel):
    id: str  # root tweet id (earliest in convo for this author)
    author_id: str
    created_at: Optional[str]
    text: str  # concatenated thread text


# ---------- X client / fields ----------

client = Client(bearer_token=os.environ["X_API_KEY"])

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


# ---------- Core helpers ----------


def process_page_with_includes(page) -> List[Post]:
    """
    Process a page of results, expanding referenced tweets from includes.
    """
    raw_data = getattr(page, "data", None) or []
    includes = getattr(page, "includes", None) or {}

    # Build a lookup of referenced tweets
    referenced_tweets_map = {}
    if "tweets" in includes:
        for tweet in includes["tweets"]:
            referenced_tweets_map[tweet.get("id")] = tweet

    posts: List[Post] = []
    for item in raw_data:
        post = Post.model_validate(item)

        # If this post references other tweets, try to expand them
        if post.referenced_tweets:
            for ref in post.referenced_tweets:
                ref_id = ref.get("id")
                if ref_id and ref_id in referenced_tweets_map:
                    post.expanded_referenced_tweet = referenced_tweets_map[ref_id]
                    break  # Just use the first one (for RTs, there's typically only one)

        posts.append(post)

    return posts


def search_author_posts(
    authors: List[str],
    query: str,
    max_results: int = 50,
    include_retweets: bool = True,
) -> List[Post]:
    """
    Search recent tweets from given authors matching `query`.
    Returns a list of Post objects (processes all pages up to max_results).
    """
    from_clause = " OR ".join(f"from:{u}" for u in authors) if authors else ""

    # Add -is:retweet filter if we don't want RTs
    rt_filter = "" if include_retweets else "-is:retweet"

    if from_clause and query:
        full_query = f"({from_clause}) ({query}) {rt_filter}".strip()
    elif from_clause:
        full_query = f"{from_clause} {rt_filter}".strip()
    else:
        full_query = f"{query} {rt_filter}".strip()

    posts: List[Post] = []

    try:
        for page in client.posts.search_recent(
            query=full_query,
            max_results=min(max_results, 100),
            tweet_fields=TWEET_FIELDS,
            expansions=EXPANSIONS,
        ):
            posts.extend(process_page_with_includes(page))

            # Process multiple pages if needed
            if len(posts) >= max_results:
                break

    except HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (429, 503):
            # rate-limited or temporary issue -> return whatever we got
            return posts
        raise

    return posts


def fetch_author_conversation(author_id: str, conv_id: str) -> List[Post]:
    """
    Fetch recent tweets by `author_id` in conversation `conv_id`.

    On 429/503, returns [] so caller can just fall back to root tweet.
    """
    if not author_id:
        return []

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


def build_thread_simple(root: Post) -> ThreadSummary:
    """
    Build a simple thread:

    - All tweets by this author in this conversation
    - Sorted by created_at
    - Concatenated full text (using note_tweet when available)
    """
    author_id = root.author_id or ""
    conv_id = root.conversation_id or root.id

    posts = fetch_author_conversation(author_id, conv_id)

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
    max_results: int = 50,
    include_retweets: bool = True,
) -> List[ThreadSummary]:
    """
    High-level entry:

    - Search recent tweets from authors matching `query`
    - Group by conversation_id
    - For each conversation, return a ThreadSummary with:
        id, author_id, created_at, text (full, untruncated)
    """
    posts = search_author_posts(
        authors,
        query=query,
        max_results=max_results,
        include_retweets=include_retweets,
    )

    # pick earliest tweet per conversation as root
    conv_to_root: Dict[str, Post] = {}
    for p in posts:
        conv_id = p.conversation_id or p.id
        current = conv_to_root.get(conv_id)
        if current is None or (p.created_at or "") < (current.created_at or ""):
            conv_to_root[conv_id] = p

    # sort roots by time for nicer printing
    roots = sorted(
        conv_to_root.values(),
        key=lambda p: p.created_at or "",
        reverse=False,
    )

    threads: List[ThreadSummary] = []
    for root in roots:
        threads.append(build_thread_simple(root))

    return threads


# ---------- Example usage ----------


def main() -> None:
    AUTHORS = [
        "iScienceLuvr",
        "arankomatsuzaki",
        "rohanpaul_ai",
        "omarsar0",
        "_akhaliq",
    ]
    QUERY = "attention"  # extra filter; use "" for "any tweet from these authors"

    # Set include_retweets=False if you want to exclude RTs entirely
    threads = get_threads_for_authors(
        AUTHORS,
        query=QUERY,
        max_results=10,
        include_retweets=True,  # Set to False to exclude RTs
    )

    for t in threads:
        print("=" * 80)
        print(f"id      : {t.id}")
        print(f"author  : {t.author_id}")
        print(f"created : {t.created_at}")
        print()
        print(t.text)
        print()


if __name__ == "__main__":
    main()
