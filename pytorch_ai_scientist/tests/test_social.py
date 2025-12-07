"""Tests for social (X search) helpers."""

from pytorch_scientist.config import XSearchConfig
from pytorch_scientist.social import MockXClient, get_threads_for_authors


def test_get_threads_with_mock_client():
    cfg = XSearchConfig(api_key=None, max_results=5)
    client = MockXClient()

    threads = get_threads_for_authors(
        authors=["author1", "author2"],
        query="attention",
        x_config=cfg,
        client=client,
    )

    assert len(threads) == 2
    assert all(t.text for t in threads)
