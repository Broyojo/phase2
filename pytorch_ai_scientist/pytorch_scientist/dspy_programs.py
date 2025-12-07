"""
DSPy signatures and modules for PyTorch Scientist.

Implements all LLM interactions through DSPy for:
- Literature summarization and gap analysis
- Idea generation and novelty checking
- Experiment summarization

This allows prompt optimization and easy provider switching.
"""

from __future__ import annotations

import json
from typing import Any

import dspy
from pydantic import BaseModel, Field

from pytorch_scientist.config import LLMConfig, LLMProvider
from pytorch_scientist.utils.logging import get_logger

logger = get_logger("dspy_programs")


# =============================================================================
# DSPy Configuration
# =============================================================================


def configure_dspy_lm(config: LLMConfig) -> dspy.LM:
    """
    Configure DSPy with the appropriate LLM backend.

    Args:
        config: LLM configuration

    Returns:
        Configured DSPy LM instance
    """
    if config.provider == LLMProvider.GROK:
        # Grok uses OpenAI-compatible API
        lm = dspy.LM(
            model=f"openai/{config.model}",
            api_key=config.grok_api_key,
            api_base=config.grok_base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.provider == LLMProvider.ANTHROPIC:
        lm = dspy.LM(
            model=f"anthropic/{config.model}",
            api_key=config.anthropic_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.provider == LLMProvider.OPENAI:
        lm = dspy.LM(
            model=f"openai/{config.model}",
            api_key=config.openai_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

    dspy.configure(lm=lm)
    logger.info(f"Configured DSPy with {config.provider.value}/{config.model}")
    return lm


# =============================================================================
# Pydantic Models for Structured Outputs
# =============================================================================


class Paper(BaseModel):
    """A research paper."""

    title: str = Field(description="Paper title")
    url: str = Field(description="URL to paper", default="")
    year: int | None = Field(description="Publication year", default=None)
    summary: str = Field(description="Brief summary of the paper")
    relevance: str = Field(description="Relevance to the research topic")


class ResearchGap(BaseModel):
    """An identified research gap."""

    description: str = Field(description="Description of the gap")
    potential_impact: str = Field(description="Potential impact if addressed")
    difficulty: str = Field(description="Estimated difficulty: easy/medium/hard")


class LiteratureSummaryOutput(BaseModel):
    """Output from literature summarization."""

    key_papers: list[Paper] = Field(description="Key papers in the field")
    open_problems: list[str] = Field(description="Open problems identified")
    unexplored_directions: list[str] = Field(description="Unexplored research directions")
    recent_trends: list[str] = Field(description="Recent trends in the field")
    research_gaps: list[ResearchGap] = Field(description="Specific research gaps")


class OptimizationIdea(BaseModel):
    """A generated optimization idea."""

    title: str = Field(description="Concise title for the idea")
    description: str = Field(description="Detailed description of the idea")
    novelty: str = Field(description="What makes this idea novel")
    implementation_sketch: str = Field(description="High-level implementation approach")
    expected_outcome: str = Field(description="Expected results and metrics")
    risk_level: str = Field(description="Risk level: low/medium/high")
    config_space: str = Field(description="Description of the config search space")
    feasibility_score: float = Field(description="Feasibility score 0-1", ge=0, le=1)
    novelty_score: float = Field(description="Novelty score 0-1", ge=0, le=1)


class NoveltyAssessment(BaseModel):
    """Assessment of an idea's novelty."""

    novelty_score: float = Field(description="Novelty score 0-1", ge=0, le=1)
    similar_works: list[str] = Field(description="Similar existing works")
    differentiation: str = Field(description="What differentiates this idea")
    explanation: str = Field(description="Explanation of the assessment")


class ExperimentSummary(BaseModel):
    """Summary of an experiment run."""

    short_summary: str = Field(description="2-3 paragraph summary")
    key_findings: list[str] = Field(description="Key findings as bullet points")
    best_config_explanation: str = Field(description="Explanation of why the best config works")
    future_work: list[str] = Field(description="Future work suggestions")
    limitations: list[str] = Field(description="Limitations of the experiment")


# =============================================================================
# DSPy Signatures
# =============================================================================


class SummarizeLiterature(dspy.Signature):
    """
    Analyze research literature and identify gaps for PyTorch optimization.

    Given raw search results from academic papers, extract key insights,
    identify open problems, and suggest unexplored research directions
    specifically in the context of PyTorch kernel and LLM optimization.
    """

    topic: str = dspy.InputField(desc="The research topic/domain")
    paper_summaries: str = dspy.InputField(desc="Raw paper summaries from search results")
    constraints: str = dspy.InputField(desc="Constraints and focus areas")

    key_papers: str = dspy.OutputField(desc="JSON list of key papers with title, summary, relevance")
    open_problems: str = dspy.OutputField(desc="JSON list of open problems")
    unexplored_directions: str = dspy.OutputField(desc="JSON list of unexplored directions")
    recent_trends: str = dspy.OutputField(desc="JSON list of recent trends")
    research_gaps: str = dspy.OutputField(desc="JSON list of research gaps with impact and difficulty")


class GenerateOptimizationIdeas(dspy.Signature):
    """
    Generate novel PyTorch optimization ideas based on literature analysis.

    Given a summary of the current research landscape, generate concrete,
    implementable ideas for PyTorch/Helion optimization that could be
    validated through config search experiments.
    """

    literature_summary: str = dspy.InputField(desc="Summary of literature and gaps")
    constraints: str = dspy.InputField(desc="Implementation constraints")
    num_ideas: int = dspy.InputField(desc="Number of ideas to generate")

    ideas: str = dspy.OutputField(
        desc="JSON array of optimization ideas with title, description, novelty, "
        "implementation_sketch, expected_outcome, risk_level, config_space, "
        "feasibility_score, novelty_score"
    )


class AssessNovelty(dspy.Signature):
    """
    Assess the novelty of an optimization idea given existing literature.

    Compare the proposed idea against existing works and determine
    how novel and differentiated it is.
    """

    idea_title: str = dspy.InputField(desc="Title of the idea")
    idea_description: str = dspy.InputField(desc="Description of the idea")
    similar_papers: str = dspy.InputField(desc="Summaries of similar existing papers")

    novelty_score: float = dspy.OutputField(desc="Novelty score from 0 to 1")
    similar_works: str = dspy.OutputField(desc="JSON list of similar works found")
    differentiation: str = dspy.OutputField(desc="What makes this idea different")
    explanation: str = dspy.OutputField(desc="Detailed explanation of assessment")


class SummarizeExperimentResult(dspy.Signature):
    """
    Generate a human-readable summary of experiment results.

    Given the idea, search trajectory, and metrics, produce a clear
    summary suitable for a hackathon demo or research report.
    """

    idea: str = dspy.InputField(desc="JSON representation of the idea")
    search_result: str = dspy.InputField(desc="JSON with configs tried, metrics, best config")
    literature_context: str = dspy.InputField(desc="Brief literature context")
    baseline_metrics: str = dspy.InputField(desc="Baseline metrics for comparison")

    short_summary: str = dspy.OutputField(desc="2-3 paragraph summary of the experiment")
    key_findings: str = dspy.OutputField(desc="JSON list of key findings")
    best_config_explanation: str = dspy.OutputField(desc="Why the best config works")
    future_work: str = dspy.OutputField(desc="JSON list of future work suggestions")
    limitations: str = dspy.OutputField(desc="JSON list of limitations")


class RefineIdea(dspy.Signature):
    """
    Refine and improve an optimization idea based on feedback.

    Take an initial idea and feedback about feasibility or gaps,
    and produce an improved version.
    """

    original_idea: str = dspy.InputField(desc="JSON of the original idea")
    feedback: str = dspy.InputField(desc="Feedback and suggestions for improvement")
    constraints: str = dspy.InputField(desc="Implementation constraints")

    refined_idea: str = dspy.OutputField(
        desc="JSON of refined idea with same structure as original"
    )
    changes_made: str = dspy.OutputField(desc="Summary of changes made")


# =============================================================================
# DSPy Modules
# =============================================================================


class LiteratureSummarizer(dspy.Module):
    """
    DSPy module for summarizing literature and identifying gaps.
    """

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(SummarizeLiterature)

    def forward(
        self,
        topic: str,
        paper_summaries: str,
        constraints: str = "",
    ) -> LiteratureSummaryOutput:
        """
        Summarize literature and identify gaps.

        Args:
            topic: Research topic
            paper_summaries: Raw paper summaries
            constraints: Optional constraints

        Returns:
            Structured literature summary
        """
        result = self.summarize(
            topic=topic,
            paper_summaries=paper_summaries,
            constraints=constraints or "Focus on PyTorch >= 2.4, transformer workloads, kernel optimization",
        )

        # Parse JSON outputs
        try:
            key_papers = json.loads(result.key_papers)
        except (json.JSONDecodeError, TypeError):
            key_papers = [{"title": "Parse error", "summary": result.key_papers, "relevance": "unknown"}]

        try:
            open_problems = json.loads(result.open_problems)
        except (json.JSONDecodeError, TypeError):
            open_problems = [result.open_problems]

        try:
            unexplored = json.loads(result.unexplored_directions)
        except (json.JSONDecodeError, TypeError):
            unexplored = [result.unexplored_directions]

        try:
            trends = json.loads(result.recent_trends)
        except (json.JSONDecodeError, TypeError):
            trends = [result.recent_trends]

        try:
            gaps = json.loads(result.research_gaps)
        except (json.JSONDecodeError, TypeError):
            gaps = [{"description": result.research_gaps, "potential_impact": "unknown", "difficulty": "medium"}]

        return LiteratureSummaryOutput(
            key_papers=[Paper(**p) if isinstance(p, dict) else Paper(title=str(p), summary="", relevance="") for p in key_papers],
            open_problems=open_problems if isinstance(open_problems, list) else [open_problems],
            unexplored_directions=unexplored if isinstance(unexplored, list) else [unexplored],
            recent_trends=trends if isinstance(trends, list) else [trends],
            research_gaps=[ResearchGap(**g) if isinstance(g, dict) else ResearchGap(description=str(g), potential_impact="unknown", difficulty="medium") for g in gaps],
        )


class IdeaGenerator(dspy.Module):
    """
    DSPy module for generating optimization ideas.
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateOptimizationIdeas)

    def forward(
        self,
        literature_summary: str,
        constraints: str,
        num_ideas: int = 5,
    ) -> list[OptimizationIdea]:
        """
        Generate optimization ideas.

        Args:
            literature_summary: Summary of literature and gaps
            constraints: Implementation constraints
            num_ideas: Number of ideas to generate

        Returns:
            List of optimization ideas
        """
        result = self.generate(
            literature_summary=literature_summary,
            constraints=constraints,
            num_ideas=num_ideas,
        )

        # Parse ideas
        try:
            ideas_data = json.loads(result.ideas)
            if not isinstance(ideas_data, list):
                ideas_data = [ideas_data]
        except (json.JSONDecodeError, TypeError):
            # Try to extract from string
            ideas_data = [{
                "title": "Generated Idea",
                "description": result.ideas,
                "novelty": "Unknown",
                "implementation_sketch": "To be determined",
                "expected_outcome": "Speedup over baseline",
                "risk_level": "medium",
                "config_space": "To be defined",
                "feasibility_score": 0.5,
                "novelty_score": 0.5,
            }]

        ideas = []
        for idea_dict in ideas_data:
            try:
                # Ensure required fields have defaults
                idea_dict.setdefault("feasibility_score", 0.5)
                idea_dict.setdefault("novelty_score", 0.5)
                ideas.append(OptimizationIdea(**idea_dict))
            except Exception as e:
                logger.warning(f"Failed to parse idea: {e}")
                continue

        return ideas


class NoveltyAssessor(dspy.Module):
    """
    DSPy module for assessing idea novelty.
    """

    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(AssessNovelty)

    def forward(
        self,
        idea_title: str,
        idea_description: str,
        similar_papers: str,
    ) -> NoveltyAssessment:
        """
        Assess the novelty of an idea.

        Args:
            idea_title: Title of the idea
            idea_description: Description of the idea
            similar_papers: Summaries of similar papers

        Returns:
            Novelty assessment
        """
        result = self.assess(
            idea_title=idea_title,
            idea_description=idea_description,
            similar_papers=similar_papers,
        )

        # Parse similar works
        try:
            similar_works = json.loads(result.similar_works)
        except (json.JSONDecodeError, TypeError):
            similar_works = [result.similar_works] if result.similar_works else []

        # Parse novelty score
        try:
            novelty_score = float(result.novelty_score)
            novelty_score = max(0.0, min(1.0, novelty_score))
        except (ValueError, TypeError):
            novelty_score = 0.5

        return NoveltyAssessment(
            novelty_score=novelty_score,
            similar_works=similar_works if isinstance(similar_works, list) else [similar_works],
            differentiation=result.differentiation,
            explanation=result.explanation,
        )


class ExperimentSummarizer(dspy.Module):
    """
    DSPy module for summarizing experiment results.
    """

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(SummarizeExperimentResult)

    def forward(
        self,
        idea: dict[str, Any] | str,
        search_result: dict[str, Any] | str,
        literature_context: str,
        baseline_metrics: dict[str, Any] | str,
    ) -> ExperimentSummary:
        """
        Summarize experiment results.

        Args:
            idea: The idea that was tested
            search_result: Search trajectory and results
            literature_context: Brief literature context
            baseline_metrics: Baseline performance metrics

        Returns:
            Experiment summary
        """
        # Convert dicts to JSON strings
        if isinstance(idea, dict):
            idea = json.dumps(idea)
        if isinstance(search_result, dict):
            search_result = json.dumps(search_result)
        if isinstance(baseline_metrics, dict):
            baseline_metrics = json.dumps(baseline_metrics)

        result = self.summarize(
            idea=idea,
            search_result=search_result,
            literature_context=literature_context,
            baseline_metrics=baseline_metrics,
        )

        # Parse lists
        try:
            key_findings = json.loads(result.key_findings)
        except (json.JSONDecodeError, TypeError):
            key_findings = [result.key_findings]

        try:
            future_work = json.loads(result.future_work)
        except (json.JSONDecodeError, TypeError):
            future_work = [result.future_work]

        try:
            limitations = json.loads(result.limitations)
        except (json.JSONDecodeError, TypeError):
            limitations = [result.limitations]

        return ExperimentSummary(
            short_summary=result.short_summary,
            key_findings=key_findings if isinstance(key_findings, list) else [key_findings],
            best_config_explanation=result.best_config_explanation,
            future_work=future_work if isinstance(future_work, list) else [future_work],
            limitations=limitations if isinstance(limitations, list) else [limitations],
        )


class IdeaRefiner(dspy.Module):
    """
    DSPy module for refining ideas based on feedback.
    """

    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(RefineIdea)

    def forward(
        self,
        original_idea: OptimizationIdea | dict[str, Any],
        feedback: str,
        constraints: str,
    ) -> OptimizationIdea:
        """
        Refine an idea based on feedback.

        Args:
            original_idea: The original idea
            feedback: Feedback for improvement
            constraints: Implementation constraints

        Returns:
            Refined idea
        """
        if isinstance(original_idea, OptimizationIdea):
            original_idea = original_idea.model_dump()

        result = self.refine(
            original_idea=json.dumps(original_idea),
            feedback=feedback,
            constraints=constraints,
        )

        try:
            refined_data = json.loads(result.refined_idea)
            # Ensure required fields
            refined_data.setdefault("feasibility_score", 0.5)
            refined_data.setdefault("novelty_score", 0.5)
            return OptimizationIdea(**refined_data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse refined idea: {e}")
            # Return original with minor updates
            original_idea["description"] = f"{original_idea.get('description', '')} (Refined: {result.changes_made})"
            return OptimizationIdea(**original_idea)


# =============================================================================
# DSPy Optimization
# =============================================================================


def create_training_examples() -> list[dspy.Example]:
    """
    Create synthetic training examples for DSPy optimization.

    Returns:
        List of training examples
    """
    examples = [
        dspy.Example(
            topic="PyTorch kernel optimization for transformers",
            paper_summaries="""
            1. FlashAttention: Fast and Memory-Efficient Exact Attention - Shows IO-aware algorithms
               can achieve 2-4x speedup on attention by reducing memory reads/writes.
            2. Triton: An Intermediate Language for Parallel Computing - Demonstrates DSL approach
               to kernel generation with automatic tuning.
            3. torch.compile and Inductor: PyTorch's native compilation stack showing 1.5-2x speedups.
            """,
            constraints="Focus on softmax and attention, PyTorch 2.4+",
            key_papers='[{"title": "FlashAttention", "summary": "IO-aware attention", "relevance": "high"}]',
            open_problems='["Optimal tiling for variable sequence lengths", "Fused attention+FFN kernels"]',
            unexplored_directions='["LLM-guided autotuning", "Multi-objective optimization"]',
            recent_trends='["Compiler-based optimization", "Kernel fusion"]',
            research_gaps='[{"description": "No systematic study of LLM-guided kernel tuning", "potential_impact": "high", "difficulty": "medium"}]',
        ).with_inputs("topic", "paper_summaries", "constraints"),

        dspy.Example(
            literature_summary="Key gaps: No LLM-guided autotuning, limited multi-objective optimization.",
            constraints="24-hour implementation, PyTorch 2.4+, transformer ops",
            num_ideas=3,
            ideas=json.dumps([{
                "title": "MCTS-Guided Helion Config Search",
                "description": "Use Monte Carlo Tree Search to efficiently explore Helion configuration space",
                "novelty": "First application of MCTS to kernel autotuning",
                "implementation_sketch": "1. Define config space 2. Implement MCTS with UCB1 3. Evaluate configs",
                "expected_outcome": "10-20% speedup over random search baseline",
                "risk_level": "medium",
                "config_space": "Tiling sizes, block dimensions, memory layouts",
                "feasibility_score": 0.8,
                "novelty_score": 0.7,
            }]),
        ).with_inputs("literature_summary", "constraints", "num_ideas"),
    ]

    return examples


def optimize_dspy_programs(
    programs: list[dspy.Module],
    examples: list[dspy.Example] | None = None,
    metric: str = "quality",
) -> list[dspy.Module]:
    """
    Optimize DSPy programs using the provided examples.

    This is a simplified optimization that demonstrates the concept.
    In production, you would use more sophisticated metrics and
    larger training sets.

    Args:
        programs: List of DSPy modules to optimize
        examples: Training examples (uses synthetic if None)
        metric: Optimization metric

    Returns:
        Optimized programs
    """
    if examples is None:
        examples = create_training_examples()

    logger.info(f"Optimizing {len(programs)} DSPy programs with {len(examples)} examples")

    # For now, return programs as-is
    # In production, you would use dspy.BootstrapFewShot or similar
    # optimizers with proper metrics

    # Example of how optimization would work:
    # from dspy.teleprompt import BootstrapFewShot
    # optimizer = BootstrapFewShot(metric=your_metric_fn)
    # optimized = optimizer.compile(program, trainset=examples)

    return programs


# =============================================================================
# Utility Functions
# =============================================================================


def get_all_programs() -> dict[str, dspy.Module]:
    """
    Get all DSPy programs as a dictionary.

    Returns:
        Dictionary mapping names to program instances
    """
    return {
        "literature_summarizer": LiteratureSummarizer(),
        "idea_generator": IdeaGenerator(),
        "novelty_assessor": NoveltyAssessor(),
        "experiment_summarizer": ExperimentSummarizer(),
        "idea_refiner": IdeaRefiner(),
    }
