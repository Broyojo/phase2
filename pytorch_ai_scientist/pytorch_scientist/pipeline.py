"""
End-to-end pipeline for PyTorch optimization research.

Orchestrates:
1. Literature discovery
2. Idea generation
3. Experiment execution
4. Summary generation
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pytorch_scientist.config import ResearchConfig, TargetOperation
from pytorch_scientist.dspy_programs import (
    ExperimentSummarizer,
    configure_dspy_lm,
)
from pytorch_scientist.experiments import ExperimentRunner, create_experiment_runner
from pytorch_scientist.ideation import Idea, generate_ideas
from pytorch_scientist.literature import LiteratureSummary, discover
from pytorch_scientist.search_engine import SearchResult, run_search
from pytorch_scientist.search_space import (
    SoftmaxConfigSpace,
    TorchCompileConfigSpace,
)
from pytorch_scientist.utils.logging import get_logger, setup_from_config
from pytorch_scientist.utils.persistence import ArtifactManager

logger = get_logger("pipeline")


def run_pytorch_scientist(config: ResearchConfig) -> dict[str, Any]:
    """
    Run the complete PyTorch Scientist pipeline.

    Args:
        config: Research configuration

    Returns:
        Dictionary with run results and artifact paths
    """
    # Set up logging
    setup_from_config(config)
    logger.info(f"Starting PyTorch Scientist run: {config.run_name}")
    logger.info(f"Domain: {config.domain}")

    # Initialize artifact manager
    artifacts = ArtifactManager(config.run_dir)

    # Track run metadata
    run_metadata = {
        "run_name": config.run_name,
        "domain": config.domain,
        "start_time": datetime.now().isoformat(),
        "config": {
            "llm_provider": config.llm.provider.value,
            "llm_model": config.llm.model,
            "search_strategy": config.search.strategy.value,
            "max_ideas": config.ideation.max_ideas,
            "max_evaluations": config.search.max_evaluations,
            "target_operation": config.experiment.target_operation.value,
        },
    }

    try:
        # =================================================================
        # PHASE 1: Literature Discovery
        # =================================================================
        logger.info("=" * 60)
        logger.info("PHASE 1: Literature Discovery")
        logger.info("=" * 60)

        literature_summary = discover(config.domain, config)

        # Save literature summary
        artifacts.save_literature_summary(literature_summary.to_dict())

        logger.info(
            f"Found {len(literature_summary.key_papers)} key papers, "
            f"{len(literature_summary.research_gaps)} research gaps"
        )

        # =================================================================
        # PHASE 2: Idea Generation
        # =================================================================
        logger.info("=" * 60)
        logger.info("PHASE 2: Idea Generation")
        logger.info("=" * 60)

        ideas = generate_ideas(literature_summary, config, check_novelty=True)

        # Save all ideas
        artifacts.save_ideas([idea.to_dict() for idea in ideas])

        if not ideas:
            logger.error("No ideas generated!")
            run_metadata["status"] = "failed"
            run_metadata["error"] = "No ideas generated"
            artifacts.save_run_metadata(run_metadata)
            return {
                "status": "failed",
                "error": "No ideas generated",
                "run_dir": str(config.run_dir),
            }

        logger.info(f"Generated {len(ideas)} ideas:")
        for i, idea in enumerate(ideas, 1):
            logger.info(
                f"  {i}. {idea.title} "
                f"(novelty={idea.novelty_score:.2f}, feasibility={idea.feasibility_score:.2f})"
            )

        # Select best idea
        selected_idea = ideas[0]  # Already sorted by combined score
        artifacts.save_selected_idea(selected_idea.to_dict())

        logger.info(f"Selected idea: {selected_idea.title}")

        # =================================================================
        # PHASE 3: Experiment Execution
        # =================================================================
        logger.info("=" * 60)
        logger.info("PHASE 3: Experiment Execution")
        logger.info("=" * 60)

        # Set up experiment runner
        experiment_runner = create_experiment_runner(
            operation=config.experiment.target_operation,
            device=config.experiment.device,
        )

        # Set up with default problem size
        experiment_runner.setup(
            batch_size=config.experiment.default_batch_size,
            seq_len=config.experiment.default_seq_len,
            hidden_dim=config.experiment.default_hidden_dim,
        )

        # Run config search
        search_result = run_search(
            idea=selected_idea,
            experiment_runner=experiment_runner,
            search_config=config.search,
            max_evaluations=config.search.max_evaluations,
        )

        # Save search results
        artifacts.save_search_history([e.to_dict() for e in search_result.history])
        artifacts.save_best_result(search_result.to_dict())

        logger.info(
            f"Search complete: "
            f"best_speedup={search_result.best_speedup:.4f}x, "
            f"best_latency={search_result.best_latency_ms:.3f}ms"
        )

        # =================================================================
        # PHASE 4: Summary Generation
        # =================================================================
        logger.info("=" * 60)
        logger.info("PHASE 4: Summary Generation")
        logger.info("=" * 60)

        # Configure DSPy for summarization
        configure_dspy_lm(config.llm)

        summarizer = ExperimentSummarizer()

        experiment_summary = summarizer(
            idea=selected_idea.to_dict(),
            search_result=search_result.to_dict(),
            literature_context=literature_summary.to_summary_string()[:2000],
            baseline_metrics=experiment_runner.get_baseline_metrics(),
        )

        # Generate markdown summary
        summary_md = _generate_markdown_summary(
            idea=selected_idea,
            search_result=search_result,
            experiment_summary=experiment_summary,
            config=config,
        )

        artifacts.save_summary(summary_md)

        # Update metadata
        run_metadata["end_time"] = datetime.now().isoformat()
        run_metadata["status"] = "completed"
        run_metadata["results"] = {
            "selected_idea": selected_idea.title,
            "best_speedup": search_result.best_speedup,
            "best_latency_ms": search_result.best_latency_ms,
            "total_evaluations": search_result.total_evaluations,
        }
        artifacts.save_run_metadata(run_metadata)

        logger.info("=" * 60)
        logger.info("Run Complete!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {config.run_dir}")
        logger.info(f"Best idea: {selected_idea.title}")
        logger.info(f"Best speedup: {search_result.best_speedup:.4f}x")

        return {
            "status": "completed",
            "run_dir": str(config.run_dir),
            "selected_idea": selected_idea.title,
            "best_speedup": search_result.best_speedup,
            "best_latency_ms": search_result.best_latency_ms,
            "best_config": search_result.best_config,
            "total_evaluations": search_result.total_evaluations,
            "summary": experiment_summary.short_summary,
        }

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        run_metadata["end_time"] = datetime.now().isoformat()
        run_metadata["status"] = "failed"
        run_metadata["error"] = str(e)
        artifacts.save_run_metadata(run_metadata)

        return {
            "status": "failed",
            "error": str(e),
            "run_dir": str(config.run_dir),
        }


def run_discovery_only(config: ResearchConfig) -> dict[str, Any]:
    """
    Run only the literature discovery and idea generation phases.

    Args:
        config: Research configuration

    Returns:
        Dictionary with discovery results
    """
    setup_from_config(config)
    logger.info(f"Starting discovery-only run: {config.run_name}")

    artifacts = ArtifactManager(config.run_dir)

    # Phase 1: Literature Discovery
    logger.info("Phase 1: Literature Discovery")
    literature_summary = discover(config.domain, config)
    artifacts.save_literature_summary(literature_summary.to_dict())

    # Phase 2: Idea Generation
    logger.info("Phase 2: Idea Generation")
    ideas = generate_ideas(literature_summary, config, check_novelty=True)
    artifacts.save_ideas([idea.to_dict() for idea in ideas])

    logger.info(f"Discovery complete: {len(ideas)} ideas generated")

    return {
        "status": "completed",
        "run_dir": str(config.run_dir),
        "num_papers": len(literature_summary.key_papers),
        "num_gaps": len(literature_summary.research_gaps),
        "num_ideas": len(ideas),
        "ideas": [
            {
                "title": idea.title,
                "novelty_score": idea.novelty_score,
                "feasibility_score": idea.feasibility_score,
            }
            for idea in ideas
        ],
    }


def _generate_markdown_summary(
    idea: Idea,
    search_result: SearchResult,
    experiment_summary: Any,
    config: ResearchConfig,
) -> str:
    """Generate a markdown summary of the experiment."""
    lines = [
        f"# PyTorch Optimization Experiment Report",
        f"",
        f"**Run Name:** {config.run_name}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Domain:** {config.domain}",
        f"",
        f"---",
        f"",
        f"## Selected Idea",
        f"",
        f"### {idea.title}",
        f"",
        f"**Description:** {idea.description}",
        f"",
        f"**Novelty:** {idea.novelty}",
        f"",
        f"**Implementation Sketch:** {idea.implementation_sketch}",
        f"",
        f"**Scores:**",
        f"- Novelty Score: {idea.novelty_score:.2f}",
        f"- Feasibility Score: {idea.feasibility_score:.2f}",
        f"- Risk Level: {idea.risk_level}",
        f"",
        f"---",
        f"",
        f"## Experiment Results",
        f"",
        f"### Performance",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best Speedup | {search_result.best_speedup:.4f}x |",
        f"| Best Latency | {search_result.best_latency_ms:.3f}ms |",
        f"| Baseline Latency | {search_result.baseline_latency_ms:.3f}ms |",
        f"| Total Evaluations | {search_result.total_evaluations} |",
        f"| Generations | {search_result.generations} |",
        f"",
        f"### Best Configuration",
        f"",
        f"```json",
        f"{json.dumps(search_result.best_config, indent=2)}",
        f"```",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"{experiment_summary.short_summary}",
        f"",
        f"### Key Findings",
        f"",
    ]

    for finding in experiment_summary.key_findings:
        lines.append(f"- {finding}")

    lines.extend([
        f"",
        f"### Best Configuration Explanation",
        f"",
        f"{experiment_summary.best_config_explanation}",
        f"",
        f"---",
        f"",
        f"## Future Work",
        f"",
    ])

    for item in experiment_summary.future_work:
        lines.append(f"- {item}")

    lines.extend([
        f"",
        f"## Limitations",
        f"",
    ])

    for item in experiment_summary.limitations:
        lines.append(f"- {item}")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## Search History Statistics",
        f"",
    ])

    stats = search_result.get_summary_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"- **{key}:** {value:.4f}")
        else:
            lines.append(f"- **{key}:** {value}")

    return "\n".join(lines)


# Convenience functions for CLI


def quick_demo(
    domain: str = "LLM guided PyTorch kernel optimization",
    max_ideas: int = 3,
    max_evaluations: int = 20,
) -> dict[str, Any]:
    """
    Run a quick demo with reduced parameters.

    Args:
        domain: Research domain
        max_ideas: Maximum ideas to generate
        max_evaluations: Maximum config evaluations

    Returns:
        Run results
    """
    config = ResearchConfig(
        domain=domain,
        run_name=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    config.ideation.max_ideas = max_ideas
    config.search.max_evaluations = max_evaluations

    return run_pytorch_scientist(config)
