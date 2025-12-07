#!/usr/bin/env python3
"""
Run the full PyTorch Scientist pipeline.

This script runs:
1. Literature discovery
2. Idea generation
3. Config search experiments
4. Summary generation

Usage:
    python scripts/run_full_pipeline.py --domain "LLM guided PyTorch kernel optimization"
    python scripts/run_full_pipeline.py --max-ideas 5 --max-evals 100
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_scientist.config import (
    LLMConfig,
    LLMProvider,
    ResearchConfig,
    SearchStrategy,
    TargetOperation,
)
from pytorch_scientist.pipeline import run_pytorch_scientist


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the PyTorch Scientist optimization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults
    python scripts/run_full_pipeline.py

    # Specify domain and limits
    python scripts/run_full_pipeline.py \\
        --domain "PyTorch attention optimization" \\
        --max-ideas 5 \\
        --max-evals 100

    # Use specific LLM provider
    python scripts/run_full_pipeline.py --llm-provider grok --llm-model grok-3-mini

    # Target specific operation
    python scripts/run_full_pipeline.py --operation attention

Environment Variables:
    EXA_API_KEY       - Exa.ai API key for literature search
    XAI_API_KEY       - Grok API key (for grok provider)
    ANTHROPIC_API_KEY - Anthropic API key (for anthropic provider)
    OPENAI_API_KEY    - OpenAI API key (for openai provider)
""",
    )

    # Domain and scope
    parser.add_argument(
        "--domain",
        type=str,
        default="LLM guided PyTorch kernel optimization",
        help="Research domain to explore (default: %(default)s)",
    )
    parser.add_argument(
        "--max-ideas",
        type=int,
        default=5,
        help="Maximum number of ideas to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=50,
        help="Maximum config evaluations (default: %(default)s)",
    )

    # LLM configuration
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["grok", "anthropic", "openai"],
        default="grok",
        help="LLM provider (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model name (default: provider-specific)",
    )

    # Experiment configuration
    parser.add_argument(
        "--operation",
        type=str,
        choices=["softmax", "gemm", "attention"],
        default="softmax",
        help="Target operation to optimize (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for experiments (default: %(default)s)",
    )

    # Search configuration
    parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["evolutionary", "mcts", "random"],
        default="evolutionary",
        help="Search strategy (default: %(default)s)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name (default: auto-generated)",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set up LLM config
    llm_provider = LLMProvider(args.llm_provider)
    llm_model = args.llm_model
    if llm_model is None:
        # Default models per provider
        default_models = {
            LLMProvider.GROK: "grok-3-mini",
            LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            LLMProvider.OPENAI: "gpt-4o",
        }
        llm_model = default_models.get(llm_provider, "grok-3-mini")

    llm_config = LLMConfig(
        provider=llm_provider,
        model=llm_model,
    )

    # Set up research config
    config = ResearchConfig(
        domain=args.domain,
        llm=llm_config,
        output_dir=args.output_dir,
        run_name=args.run_name,
        log_level="DEBUG" if args.verbose else "INFO",
        verbose=args.verbose,
    )

    # Update sub-configs
    config.ideation.max_ideas = args.max_ideas
    config.search.max_evaluations = args.max_evals
    config.search.strategy = SearchStrategy(args.search_strategy)
    config.experiment.target_operation = TargetOperation(args.operation)
    config.experiment.device = args.device

    print(f"PyTorch Scientist - Full Pipeline")
    print(f"=" * 50)
    print(f"Domain: {config.domain}")
    print(f"LLM: {config.llm.provider.value}/{config.llm.model}")
    print(f"Target Operation: {config.experiment.target_operation.value}")
    print(f"Max Ideas: {config.ideation.max_ideas}")
    print(f"Max Evaluations: {config.search.max_evaluations}")
    print(f"Search Strategy: {config.search.strategy.value}")
    print(f"Output: {config.run_dir}")
    print(f"=" * 50)
    print()

    # Run pipeline
    result = run_pytorch_scientist(config)

    # Print results
    print()
    print(f"=" * 50)
    print(f"Results")
    print(f"=" * 50)

    if result["status"] == "completed":
        print(f"Status: SUCCESS")
        print(f"Selected Idea: {result['selected_idea']}")
        print(f"Best Speedup: {result['best_speedup']:.4f}x")
        print(f"Best Latency: {result['best_latency_ms']:.3f}ms")
        print(f"Total Evaluations: {result['total_evaluations']}")
        print(f"Run Directory: {result['run_dir']}")
        print()
        print("Summary:")
        print(result.get("summary", "No summary generated"))
        return 0
    else:
        print(f"Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Run Directory: {result['run_dir']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
