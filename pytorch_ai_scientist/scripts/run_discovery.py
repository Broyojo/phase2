#!/usr/bin/env python3
"""
Run only the literature discovery and idea generation phases.

This script is useful for:
- Exploring research gaps without running experiments
- Quick ideation sessions
- Testing the literature and ideation pipeline

Usage:
    python scripts/run_discovery.py --domain "PyTorch kernel optimization"
    python scripts/run_discovery.py --max-ideas 10
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_scientist.config import (
    LLMConfig,
    LLMProvider,
    ResearchConfig,
)
from pytorch_scientist.pipeline import run_discovery_only


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run literature discovery and idea generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--domain",
        type=str,
        default="LLM guided PyTorch kernel optimization",
        help="Research domain to explore",
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
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure LLM
    llm_provider = LLMProvider(args.llm_provider)
    llm_model = args.llm_model or {
        LLMProvider.GROK: "grok-3-mini",
        LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
        LLMProvider.OPENAI: "gpt-4o",
    }.get(llm_provider, "grok-3-mini")

    config = ResearchConfig(
        domain=args.domain,
        llm=LLMConfig(provider=llm_provider, model=llm_model),
        output_dir=args.output_dir,
        log_level="DEBUG" if args.verbose else "INFO",
    )
    config.ideation.max_ideas = args.max_ideas

    if not args.json:
        print(f"PyTorch Scientist - Discovery Mode")
        print(f"=" * 50)
        print(f"Domain: {config.domain}")
        print(f"LLM: {config.llm.provider.value}/{config.llm.model}")
        print(f"Max Ideas: {config.ideation.max_ideas}")
        print(f"=" * 50)
        print()

    # Run discovery
    result = run_discovery_only(config)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print(f"=" * 50)
        print(f"Discovery Results")
        print(f"=" * 50)
        print(f"Status: {result['status']}")
        print(f"Papers Found: {result['num_papers']}")
        print(f"Research Gaps: {result['num_gaps']}")
        print(f"Ideas Generated: {result['num_ideas']}")
        print(f"Run Directory: {result['run_dir']}")
        print()

        if result.get("ideas"):
            print("Generated Ideas:")
            print("-" * 40)
            for i, idea in enumerate(result["ideas"], 1):
                print(
                    f"{i}. {idea['title']}"
                    f" (novelty={idea['novelty_score']:.2f}, "
                    f"feasibility={idea['feasibility_score']:.2f})"
                )

    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
