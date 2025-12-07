import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from xdk import Client


class Post(BaseModel):
    id: str
    text: str
    edit_history_tweet_ids: Optional[List[str]] = None

    created_at: Optional[str] = None
    author_id: Optional[str] = None
    conversation_id: Optional[str] = None

    referenced_tweets: Optional[List[Dict[str, Any]]] = None
    entities: Optional[Dict[str, Any]] = None
    geo: Optional[Dict[str, Any]] = None
    lang: Optional[str] = None
    public_metrics: Optional[Dict[str, Any]] = None
    possibly_sensitive: Optional[bool] = None
    reply_settings: Optional[str] = None
    source: Optional[str] = None
    withheld: Optional[Dict[str, Any]] = None


THREAD_TWEET_FIELDS = [
    "id",
    "text",
    "edit_history_tweet_ids",
    "author_id",
    "created_at",
    "conversation_id",
    "referenced_tweets",
    "public_metrics",
]

client = Client(bearer_token=os.environ["X_API_KEY"])


def get_linear_thread_text(tweet_id: str, use_full_archive: bool = True) -> str:
    # 1) Get the starting tweet (root)
    resp = client.posts.get_by_id(
        id=tweet_id,
        tweet_fields=["id", "text", "author_id", "conversation_id", "created_at"],
    )
    root = Post.model_validate(resp.data)

    root_id = root.id
    author_id = root.author_id
    conv_id = root.conversation_id or root_id

    # 2) Fetch all tweets in that conversation *by that author*
    query = f"conversation_id:{conv_id} from:{author_id} -is:retweet"

    posts: List[Post] = []

    search_fn = (
        client.posts.search_all if use_full_archive else client.posts.search_recent
    )

    for page in search_fn(
        query=query,
        max_results=100,
        tweet_fields=THREAD_TWEET_FIELDS,
    ):
        if not page.data:
            break
        posts.extend(Post.model_validate(p) for p in page.data)

    # Include the root explicitly in case it wasn't in the search results
    if not any(p.id == root_id for p in posts):
        posts.append(root)

    # 3) Sort oldest → newest
    posts.sort(key=lambda p: p.created_at or "")

    # 4) Build *linear* chain: only tweets that reply to something already in chain
    keep_ids = {root_id}
    chain: List[Post] = []

    for p in posts:
        if p.id == root_id:
            chain.append(p)
            continue

        parent_id = None
        if p.referenced_tweets:
            for ref in p.referenced_tweets:
                if ref.get("type") == "replied_to":
                    parent_id = ref.get("id")
                    break

        # Keep only if this tweet is a reply to something already in the chain
        if parent_id in keep_ids:
            keep_ids.add(p.id)
            chain.append(p)

    # 5) Concatenate text
    return "\n\n".join(p.text for p in chain if p.text)


print(get_linear_thread_text("1997169129103479280"))
