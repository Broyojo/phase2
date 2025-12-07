import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from xdk import Client


class Post(BaseModel):
    # default root fields
    id: str
    text: str
    edit_history_tweet_ids: Optional[List[str]] = None

    # common fields
    created_at: Optional[str] = None
    author_id: Optional[str] = None

    # all other tweet.fields from the data dictionary
    article: Optional[Dict[str, Any]] = None
    attachments: Optional[Dict[str, Any]] = None
    card_uri: Optional[str] = None
    community_id: Optional[str] = None
    context_annotations: Optional[List[Dict[str, Any]]] = None
    conversation_id: Optional[str] = None
    display_text_range: Optional[List[int]] = None
    edit_controls: Optional[Dict[str, Any]] = None
    entities: Optional[Dict[str, Any]] = None
    geo: Optional[Dict[str, Any]] = None
    in_reply_to_user_id: Optional[str] = None
    lang: Optional[str] = None
    non_public_metrics: Optional[Dict[str, Any]] = None
    note_tweet: Optional[Dict[str, Any]] = None
    organic_metrics: Optional[Dict[str, Any]] = None
    possibly_sensitive: Optional[bool] = None
    promoted_metrics: Optional[Dict[str, Any]] = None
    public_metrics: Optional[Dict[str, Any]] = None
    referenced_tweets: Optional[List[Dict[str, Any]]] = None
    reply_settings: Optional[str] = None
    withheld: Optional[Dict[str, Any]] = None
    scopes: Optional[Dict[str, Any]] = None
    media_metadata: Optional[List[Dict[str, Any]]] = None


# Restrict results to these usernames (no '@')
USERS = ["iScienceLuvr", "arankomatsuzaki", "rohanpaul_ai"]

keyword_query = "attention"
from_clause = " OR ".join(f"from:{u}" for u in USERS)
query = f"({from_clause}) ({keyword_query})"

client = Client(bearer_token=os.environ["X_API_KEY"])

TWEET_FIELDS_SAFE = [
    "id",
    "text",
    "edit_history_tweet_ids",
    "author_id",
    "created_at",
    "conversation_id",
    "in_reply_to_user_id",
    "referenced_tweets",
    "entities",
    "geo",
    "lang",
    "public_metrics",  # counts you are allowed to see
    "possibly_sensitive",
    "reply_settings",
    "source",
    "withheld",
]

all_posts: list[Post] = []
for page in client.posts.search_all(
    query=query,
    max_results=10,
    tweet_fields=TWEET_FIELDS_SAFE,
):
    all_posts.extend([Post.model_validate(data) for data in page.data])
    print(f"Fetched {len(page.data)} Posts (total: {len(all_posts)})")
    break

print(f"Total tweets: {len(all_posts)}")
for post in all_posts:
    print(post)
