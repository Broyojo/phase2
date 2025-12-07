#!/usr/bin/env python3
"""
CLI entry point for PyTorch Scientist.

Provides subcommands for:
- Full pipeline
- Discovery only
- Quick demo
"""

import argparse
import sys
from pathlib import Path

from pytorch_scientist.config import (
    LLMConfig,
    LLMProvider,
    ResearchConfig,
    SearchStrategy,
    TargetOperation,
)
from pytorch_scientist.pipeline import (
    quick_demo,
    run_discovery_only,
    run_pytorch_scientist,
)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pytorch-scientist",
        description="AI Scientist for PyTorch Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
    run      - Run full pipeline (discovery + experiments)
    discover - Run only literature discovery and idea generation
    demo     - Run a quick demo with reduced parameters

Examples:
    # Full pipeline
    pytorch-scientist run --domain "PyTorch attention optimization"

    # Discovery only
    pytorch-scientist discover --max-ideas 10

    # Quick demo
    pytorch-scientist demo

Environment Variables:
    EXA_API_KEY       - Exa.ai API key
    XAI_API_KEY       - Grok API key
    ANTHROPIC_API_KEY - Anthropic API key
    OPENAI_API_KEY    - OpenAI API key
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    _add_common_args(run_parser)
    run_parser.add_argument(
        "--max-evals",
        type=int,
        default=50,
        help="Maximum config evaluations",
    )
    run_parser.add_argument(
        "--operation",
        type=str,
        choices=["softmax", "gemm", "attention"],
        default="softmax",
        help="Target operation",
    )
    run_parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["evolutionary", "mcts", "random"],
        default="evolutionary",
        help="Search strategy",
    )
    run_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for experiments",
    )

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Run discovery only")
    _add_common_args(discover_parser)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick demo")
    demo_parser.add_argument(
        "--domain",
        type=str,
        default="LLM guided PyTorch kernel optimization",
        help="Research domain",
    )
    demo_parser.add_argument(
        "--max-ideas",
        type=int,
        default=3,
        help="Maximum ideas",
    )
    demo_parser.add_argument(
        "--max-evals",
        type=int,
        default=20,
        help="Maximum evaluations",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        return _run_full(args)
    elif args.command == "discover":
        return _run_discover(args)
    elif args.command == "demo":
        return _run_demo(args)
    else:
        parser.print_help()
        return 1


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    parser.add_argument(
        "--domain",
        type=str,
        default="LLM guided PyTorch kernel optimization",
        help="Research domain",
    )
    parser.add_argument(
        "--max-ideas",
        type=int,
        default=5,
        help="Maximum ideas to generate",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["grok", "anthropic", "openai"],
        default="grok",
        help="LLM provider",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--x-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fetching X posts to augment literature (default: enabled)",
    )
    parser.add_argument(
        "--x-authors",
        type=str,
        default=None,
        help="Comma-separated X handles to search (without @). Overrides authors_file if provided.",
    )
    parser.add_argument(
        "--x-authors-file",
        type=Path,
        default=None,
        help="Path to newline-separated X handles file (without @).",
    )
    parser.add_argument(
        "--x-query",
        type=str,
        default=None,
        help="Keyword filter for X search (defaults to domain).",
    )
    parser.add_argument(
        "--x-max-posts",
        type=int,
        default=None,
        help="Max tweets to fetch from X (default 50).",
    )
    parser.add_argument(
        "--x-no-retweets",
        action="store_true",
        help="Exclude retweets from X search results.",
    )


def _get_llm_config(args: argparse.Namespace) -> LLMConfig:
    """Create LLM config from args."""
    provider = LLMProvider(args.llm_provider)
    model = args.llm_model or {
        LLMProvider.GROK: "grok-4-1-fast-reasoning",
        LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
        LLMProvider.OPENAI: "gpt-4o",
    }.get(provider, "grok-4-1-fast-reasoning")

    return LLMConfig(provider=provider, model=model)


def _run_full(args: argparse.Namespace) -> int:
    """Run full pipeline."""
    config = ResearchConfig(
        domain=args.domain,
        llm=_get_llm_config(args),
        output_dir=args.output_dir,
        log_level="DEBUG" if args.verbose else "INFO",
    )
    config.ideation.max_ideas = args.max_ideas
    config.search.max_evaluations = args.max_evals
    config.search.strategy = SearchStrategy(args.search_strategy)
    config.experiment.target_operation = TargetOperation(args.operation)
    config.experiment.device = args.device

    # X config
    config.x.enabled = args.x_enabled
    if args.x_authors:
        config.x.authors = [
            h.strip().lstrip("@") for h in args.x_authors.split(",") if h.strip()
        ]
    config.x.authors_file = args.x_authors_file
    if args.x_query:
        config.x.query = args.x_query
    if args.x_max_posts:
        config.x.max_results = args.x_max_posts
    config.x.include_retweets = not args.x_no_retweets

    result = run_pytorch_scientist(config)

    if result["status"] == "completed":
        print(f"\nSuccess! Best speedup: {result['best_speedup']:.4f}x")
        print(f"Results: {result['run_dir']}")
        return 0
    else:
        print(f"\nFailed: {result.get('error', 'Unknown error')}")
        return 1


def _run_discover(args: argparse.Namespace) -> int:
    """Run discovery only."""
    config = ResearchConfig(
        domain=args.domain,
        llm=_get_llm_config(args),
        output_dir=args.output_dir,
        log_level="DEBUG" if args.verbose else "INFO",
    )
    config.ideation.max_ideas = args.max_ideas

    config.x.enabled = args.x_enabled
    if args.x_authors:
        config.x.authors = [
            h.strip().lstrip("@") for h in args.x_authors.split(",") if h.strip()
        ]
    config.x.authors_file = args.x_authors_file
    if args.x_query:
        config.x.query = args.x_query
    if args.x_max_posts:
        config.x.max_results = args.x_max_posts
    config.x.include_retweets = not args.x_no_retweets

    result = run_discovery_only(config)

    if result["status"] == "completed":
        print(f"\nDiscovery complete: {result['num_ideas']} ideas generated")
        print(f"Results: {result['run_dir']}")
        return 0
    else:
        return 1


def _run_demo(args: argparse.Namespace) -> int:
    """Run quick demo."""
    print("Running quick demo...")
    result = quick_demo(
        domain=args.domain,
        max_ideas=args.max_ideas,
        max_evaluations=args.max_evals,
    )

    if result["status"] == "completed":
        print(f"\nDemo complete! Best speedup: {result['best_speedup']:.4f}x")
        return 0
    else:
        print(f"\nDemo failed: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
