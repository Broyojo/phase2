"""
Literature discovery using Exa.ai API.

Provides paper search, SOTA method discovery, and research gap identification
for PyTorch kernel optimization research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pytorch_scientist.config import ExaConfig, LLMConfig, ResearchConfig, XSearchConfig
from pytorch_scientist.dspy_programs import (
    LiteratureSummarizer,
    LiteratureSummaryOutput,
    Paper,
    ResearchGap,
    configure_dspy_lm,
)
from pytorch_scientist.social import ThreadSummary, get_threads_for_authors
from pytorch_scientist.utils.logging import get_logger

logger = get_logger("literature")


@dataclass
class ExaSearchResult:
    """A single search result from Exa."""

    title: str
    url: str
    score: float
    published_date: str | None = None
    author: str | None = None
    text: str | None = None
    highlights: list[str] = field(default_factory=list)

    @classmethod
    def from_exa_result(cls, result: Any) -> "ExaSearchResult":
        """Create from Exa API result object."""
        return cls(
            title=getattr(result, "title", "Unknown"),
            url=getattr(result, "url", ""),
            score=getattr(result, "score", 0.0),
            published_date=getattr(result, "published_date", None),
            author=getattr(result, "author", None),
            text=getattr(result, "text", None),
            highlights=getattr(result, "highlights", []),
        )

    def to_summary_string(self) -> str:
        """Convert to a summary string for LLM processing."""
        MAX_TEXT_CHARS = 15000  # keep rich context but avoid eating full model window
        parts = [f"Title: {self.title}"]
        if self.url:
            parts.append(f"URL: {self.url}")
        if self.published_date:
            parts.append(f"Published: {self.published_date}")
        if self.text:
            text = self.text if len(self.text) <= MAX_TEXT_CHARS else self.text[:MAX_TEXT_CHARS] + "..."
            parts.append(f"Abstract/Summary: {text}")
        if self.highlights:
            parts.append(f"Key points: {'; '.join(self.highlights[:3])}")
        return "\n".join(parts)


@dataclass
class LiteratureSummary:
    """
    Complete literature summary including papers, gaps, and trends.
    """

    domain: str
    key_papers: list[Paper]
    open_problems: list[str]
    unexplored_directions: list[str]
    recent_trends: list[str]
    research_gaps: list[ResearchGap]
    raw_search_results: list[ExaSearchResult] = field(default_factory=list)
    x_threads: list[ThreadSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain": self.domain,
            "key_papers": [p.model_dump() for p in self.key_papers],
            "open_problems": self.open_problems,
            "unexplored_directions": self.unexplored_directions,
            "recent_trends": self.recent_trends,
            "research_gaps": [g.model_dump() for g in self.research_gaps],
            "num_papers_analyzed": len(self.raw_search_results),
            "num_x_threads": len(self.x_threads),
            "x_threads": [t.to_dict() for t in self.x_threads],
        }

    def to_summary_string(self) -> str:
        """Convert to a string summary for use in ideation."""
        parts = [
            f"Domain: {self.domain}",
            "",
            "Key Papers:",
        ]
        for paper in self.key_papers[:5]:
            parts.append(f"  - {paper.title}: {paper.summary}")

        parts.extend([
            "",
            "Open Problems:",
        ])
        for problem in self.open_problems[:5]:
            parts.append(f"  - {problem}")

        parts.extend([
            "",
            "Research Gaps:",
        ])
        for gap in self.research_gaps[:5]:
            parts.append(f"  - {gap.description} (Impact: {gap.potential_impact})")

        parts.extend([
            "",
            "Recent Trends:",
        ])
        for trend in self.recent_trends[:5]:
            parts.append(f"  - {trend}")

        if self.x_threads:
            parts.extend([
                "",
                "Recent X Threads:",
            ])
            for thread in self.x_threads:
                parts.append(f"---\n{thread.to_summary_string()}")

        return "\n".join(parts)


class LiteratureDiscovery:
    """
    Literature discovery and gap analysis using Exa.ai and DSPy.

    Combines Exa's semantic search with LLM-powered analysis
    to identify research opportunities in PyTorch optimization.
    """

    # Default queries for SOTA methods
    SOTA_QUERIES = [
        "PyTorch kernel optimization techniques 2024",
        "transformer attention optimization CUDA",
        "torch.compile performance optimization",
        "Triton kernel autotuning",
        "FlashAttention implementation details",
        "LLM inference optimization GPU",
    ]

    def __init__(
        self,
        exa_config: ExaConfig | None = None,
        llm_config: LLMConfig | None = None,
        x_config: XSearchConfig | None = None,
    ):
        """
        Initialize literature discovery.

        Args:
            exa_config: Exa API configuration
            llm_config: LLM configuration for DSPy
        """
        self.exa_config = exa_config or ExaConfig()
        self.llm_config = llm_config or LLMConfig()
        self.x_config = x_config or XSearchConfig()

        # Initialize Exa client
        self._exa_client = None

        # Initialize DSPy summarizer
        self.summarizer = LiteratureSummarizer()

        logger.info("Initialized LiteratureDiscovery")

    @property
    def exa_client(self) -> Any:
        """Lazy initialization of Exa client."""
        if self._exa_client is None:
            try:
                from exa_py import Exa

                if not self.exa_config.api_key:
                    raise ValueError("EXA_API_KEY not set")

                self._exa_client = Exa(api_key=self.exa_config.api_key)
                logger.info("Initialized Exa client")
            except ImportError:
                logger.warning("exa_py not installed, using mock client")
                self._exa_client = MockExaClient()

        return self._exa_client

    def search_papers(
        self,
        query: str,
        num_results: int | None = None,
        start_date: str | None = None,
    ) -> list[ExaSearchResult]:
        """
        Search for papers using Exa.

        Args:
            query: Search query
            num_results: Number of results (uses config default if None)
            start_date: Optional start date filter (YYYY-MM-DD)

        Returns:
            List of search results
        """
        num_results = num_results or self.exa_config.num_results

        logger.info(f"Searching papers: '{query}' (n={num_results})")

        try:
            # Exa search with research paper focus
            search_kwargs: dict[str, Any] = {
                "query": query,
                "num_results": num_results,
                "type": "auto",
                "include_domains": self.exa_config.include_domains,
            }

            if start_date:
                search_kwargs["start_published_date"] = start_date

            # Perform search
            response = self.exa_client.search(
                **search_kwargs,
                contents={"text": {"maxCharacters": 10000}},
            )

            results = [
                ExaSearchResult.from_exa_result(r)
                for r in response.results
            ]

            logger.info(f"Found {len(results)} papers")
            return results

        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return []

    def get_sota_methods(
        self,
        custom_queries: list[str] | None = None,
    ) -> list[ExaSearchResult]:
        """
        Get state-of-the-art methods from multiple queries.

        Args:
            custom_queries: Optional custom queries (uses defaults if None)

        Returns:
            Combined list of search results
        """
        queries = custom_queries or self.SOTA_QUERIES
        all_results: list[ExaSearchResult] = []
        seen_urls: set[str] = set()

        for query in queries:
            results = self.search_papers(query, num_results=5)
            for r in results:
                if r.url not in seen_urls:
                    all_results.append(r)
                    seen_urls.add(r.url)

        logger.info(f"Total unique papers from SOTA queries: {len(all_results)}")
        return all_results

    def find_research_gaps(
        self,
        topic: str,
        existing_results: list[ExaSearchResult] | None = None,
    ) -> LiteratureSummaryOutput:
        """
        Analyze literature to find research gaps.

        Uses DSPy to process search results and identify opportunities.

        Args:
            topic: Research topic
            existing_results: Optional pre-fetched results

        Returns:
            Structured analysis of gaps and opportunities
        """
        # Configure DSPy
        configure_dspy_lm(self.llm_config)

        # Get papers if not provided
        if existing_results is None:
            existing_results = self.search_papers(topic, num_results=15)

        if not existing_results:
            logger.warning("No search results, using fallback analysis")
            return self._fallback_gap_analysis(topic)

        # Format papers for LLM
        paper_summaries = "\n\n---\n\n".join([
            r.to_summary_string() for r in existing_results
        ])

        # Run DSPy summarizer
        constraints = """
        Focus on:
        - PyTorch >= 2.4 and torch.compile
        - Triton and Helion DSL for kernel optimization
        - Transformer-style workloads (attention, GEMM, softmax, layernorm)
        - Automated/LLM-guided optimization approaches
        - Config search and autotuning methods
        """

        result = self.summarizer(
            topic=topic,
            paper_summaries=paper_summaries,
            constraints=constraints,
        )

        logger.info(
            f"Gap analysis complete: {len(result.key_papers)} papers, "
            f"{len(result.research_gaps)} gaps identified"
        )

        return result

    def discover(self, domain: str) -> LiteratureSummary:
        """
        Full literature discovery pipeline.

        Combines search, SOTA analysis, and gap identification.

        Args:
            domain: Research domain string

        Returns:
            Complete literature summary
        """
        logger.info(f"Starting literature discovery for: {domain}")

        # Search for relevant papers
        main_results = self.search_papers(domain, num_results=10)

        # Get SOTA methods
        sota_results = self.get_sota_methods()

        # Combine results
        all_results = main_results + sota_results
        seen_urls: set[str] = set()
        unique_results: list[ExaSearchResult] = []
        for r in all_results:
            if r.url not in seen_urls:
                unique_results.append(r)
                seen_urls.add(r.url)

        # Analyze gaps
        analysis = self.find_research_gaps(domain, unique_results)

        # Fetch X threads if enabled
        x_threads: list[ThreadSummary] = []
        if self.x_config.enabled:
            authors = self.x_config.resolved_authors()
            query = self.x_config.query or domain
            if self.x_config.api_key or authors:
                try:
                    x_threads = get_threads_for_authors(
                        authors,
                        query=query,
                        x_config=self.x_config,
                    )
                    logger.info(f"Found {len(x_threads)} X threads")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"X search failed, continuing without: {e}")
            else:
                logger.info("X search skipped: no API key/authors configured")

        # Create complete summary
        summary = LiteratureSummary(
            domain=domain,
            key_papers=analysis.key_papers,
            open_problems=analysis.open_problems,
            unexplored_directions=analysis.unexplored_directions,
            recent_trends=analysis.recent_trends,
            research_gaps=analysis.research_gaps,
            raw_search_results=unique_results,
            x_threads=x_threads,
        )

        logger.info(
            f"Discovery complete: {len(summary.key_papers)} key papers, "
            f"{len(summary.research_gaps)} research gaps"
        )

        return summary

    def _fallback_gap_analysis(self, topic: str) -> LiteratureSummaryOutput:
        """Provide fallback analysis when search fails."""
        return LiteratureSummaryOutput(
            key_papers=[
                Paper(
                    title="FlashAttention: Fast and Memory-Efficient Exact Attention",
                    url="https://arxiv.org/abs/2205.14135",
                    year=2022,
                    summary="IO-aware attention algorithm achieving 2-4x speedup",
                    relevance="Foundational work on attention optimization",
                ),
                Paper(
                    title="Triton: An Intermediate Language for Parallel Computing",
                    url="https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf",
                    year=2019,
                    summary="DSL for writing efficient GPU kernels",
                    relevance="Basis for Helion and custom kernel development",
                ),
            ],
            open_problems=[
                "Automated discovery of optimal kernel configurations",
                "LLM-guided optimization for kernel autotuning",
                "Multi-objective optimization balancing speed and memory",
                "Generalizing FlashAttention patterns to other operations",
            ],
            unexplored_directions=[
                "Using LLMs to generate and evaluate kernel implementations",
                "MCTS-based search over kernel configuration spaces",
                "Transfer learning for kernel performance prediction",
            ],
            recent_trends=[
                "Compiler-based optimization (torch.compile, Inductor)",
                "Kernel fusion for reduced memory bandwidth",
                "Quantization-aware kernel optimization",
            ],
            research_gaps=[
                ResearchGap(
                    description="No systematic study of LLM-guided kernel autotuning",
                    potential_impact="Could significantly reduce manual tuning effort",
                    difficulty="medium",
                ),
                ResearchGap(
                    description="Limited multi-objective optimization for kernels",
                    potential_impact="Enable latency-memory-accuracy trade-offs",
                    difficulty="hard",
                ),
            ],
        )


class MockExaClient:
    """
    Mock Exa client for testing and when API is unavailable.
    """

    def search_and_contents(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any,
    ) -> Any:
        """Return mock search results."""
        logger.info(f"MockExaClient: Searching '{query}'")

        # Create mock results
        class MockResult:
            def __init__(self, title: str, url: str, text: str):
                self.title = title
                self.url = url
                self.text = text
                self.score = 0.9
                self.published_date = "2024-01-01"
                self.author = "Research Team"
                self.highlights = ["Key finding 1", "Key finding 2"]

        class MockResponse:
            def __init__(self, results: list[MockResult]):
                self.results = results

        mock_papers = [
            MockResult(
                "FlashAttention-2: Faster Attention with Better Parallelism",
                "https://arxiv.org/abs/2307.08691",
                "FlashAttention-2 improves upon FlashAttention by optimizing parallelism and work partitioning. "
                "Achieves 2x speedup over FlashAttention on A100 GPUs.",
            ),
            MockResult(
                "torch.compile: Making PyTorch Faster",
                "https://pytorch.org/blog/torch-compile",
                "torch.compile provides automatic optimization of PyTorch code through TorchDynamo and Inductor. "
                "Achieves 1.5-2x speedups on various models.",
            ),
            MockResult(
                "Triton: An Intermediate Language for Parallel Programming",
                "https://triton-lang.org/",
                "Triton provides a Python-like DSL for writing efficient GPU kernels. "
                "Enables kernel fusion and automatic memory optimization.",
            ),
            MockResult(
                "Efficient Transformers: A Survey",
                "https://arxiv.org/abs/2009.06732",
                "Comprehensive survey of efficient transformer architectures and optimization techniques. "
                "Covers attention approximations, kernel methods, and sparsity.",
            ),
            MockResult(
                "MCTS for Hyperparameter Optimization",
                "https://arxiv.org/example",
                "Monte Carlo Tree Search applied to hyperparameter optimization. "
                "Shows promising results for structured search spaces.",
            ),
        ]

        return MockResponse(mock_papers[:num_results])


def discover(
    domain: str,
    config: ResearchConfig | None = None,
) -> LiteratureSummary:
    """
    Convenience function for literature discovery.

    Args:
        domain: Research domain string
        config: Optional research configuration

    Returns:
        Literature summary
    """
    if config is None:
        config = ResearchConfig()

    discovery = LiteratureDiscovery(
        exa_config=config.exa,
        llm_config=config.llm,
        x_config=config.x,
    )

    return discovery.discover(domain)
