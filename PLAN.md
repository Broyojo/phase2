# Plan: Integrate X (Twitter) search into PyTorch Scientist

## Goal
Extend the literature discovery phase so it gathers both academic papers (Exa) and X posts relevant to the research domain, then feeds that combined context into ideation and saved artifacts.

## Current State (quick notes)
- Literature discovery lives in `pytorch_scientist/literature.py` and only queries Exa.
- `LiteratureSummary` drives both `ideation.AIScientistIdeation.build_workshop_context` and `LiteratureSummary.to_summary_string()` for Grok idea generation.
- Config only has `ExaConfig`; no awareness of X API.
- New `x.py` has production-ready helpers built on `xdk`: Post model with `get_full_text`, RT expansion via includes, conversation fetching, and `get_threads_for_authors` to return full-thread text per conversation.
- `following.txt` lists candidate handles we could use as defaults.

## Plan
1) **Config & wiring**
   - Add `XSearchConfig` to `config.py` (api_key from `X_API_KEY`, enabled flag, authors list/file path, query keyword, max_results, include_retweets, recent vs full-archive toggle). Attach to `ResearchConfig` so pipeline + CLI can use it.
   - Update `pytorch_ai_scientist/pyproject.toml` to include `xdk`; document `X_API_KEY` in README/CLI help and add CLI switches like `--x-enabled/--no-x`, `--x-authors-file`, `--x-max-posts`.

2) **Data model for X posts**
   - Reuse the `Post` and `ThreadSummary` shapes from `x.py` (add URL/permalink if available) and add `to_summary_string()` for LLM prompts plus JSON-friendly `.to_dict()`.

3) **X client + search helper**
   - Move `x.py` logic into a reusable module (e.g., `pytorch_scientist/social.py`) with lazy `xdk.Client` init and a `MockXClient` for tests.
   - Keep `process_page_with_includes`, `search_author_posts`, `fetch_author_conversation`, `get_threads_for_authors`; add optional `search_all` path when `use_full_archive=True`.
   - Allow authors list to come from config or a file (e.g., `following.txt`), and enforce a safe tweet_fields/expansions allowlist.

4) **Integrate into discovery flow**
   - Extend `LiteratureDiscovery.discover()` to fetch X threads via `get_threads_for_authors` using domain keyword, honoring config limits; skip cleanly when disabled or missing API key.
   - Add optional `start_date`/`max_results` and recency mode; handle rate limits by returning partial results (as `x.py` does).
   - Store threads in `LiteratureSummary` (e.g., `x_threads` plus counts). Ensure `to_dict()` and persistence include them without breaking existing consumers.

5) **Summarization for ideation**
   - Keep Exa paper gap analysis as-is, but inject X signals: summarize top N threads (title/author/date + first 280 chars of `get_full_text()`).
   - Update `to_summary_string()` and `build_workshop_context()` to add a concise "Recent X chatter" section; cap tokens to avoid prompt bloat and optionally run a mini summarizer for "Social trends" bullets.

6) **CLI/Docs adjustments**
   - Document `X_API_KEY`, authors file option, and the new flags; update README + CLI help + quick-start to mention mixed paper+X discovery.

7) **Tests**
   - Add unit tests for `XSearchConfig` env loading, authors-file parsing, `get_threads_for_authors` with `MockXClient`, and summary string injection limits.
   - Update existing literature/config tests to tolerate optional `x_threads` while keeping legacy behaviors intact.

## Risks / Considerations
- Keep prompts concise: cap number of posts and summarize to avoid LLM context bloat.
- Ensure graceful degradation when `X_API_KEY` is missing (skip X search, leave existing behavior intact).
- Avoid breaking saved artifact schema; include clear versioning or optional fields.
