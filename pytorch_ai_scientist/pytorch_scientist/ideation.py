"""
Idea generation for PyTorch optimization.

Combines:
- AI Scientist patterns for structured ideation
- Grok (via OpenAI-compatible API) for generation
- DSPy for prompt optimization
- Exa for novelty checking
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from pytorch_scientist.config import (
    ExaConfig,
    IdeationConfig,
    LLMConfig,
    LLMProvider,
    ResearchConfig,
)
from pytorch_scientist.dspy_programs import (
    IdeaGenerator,
    IdeaRefiner,
    NoveltyAssessor,
    OptimizationIdea,
    configure_dspy_lm,
)
from pytorch_scientist.literature import (
    ExaSearchResult,
    LiteratureDiscovery,
    LiteratureSummary,
)
from pytorch_scientist.utils.logging import get_logger

logger = get_logger("ideation")


@dataclass
class Idea:
    """
    A generated optimization idea.
    """

    title: str
    description: str
    novelty: str
    implementation_sketch: str
    expected_outcome: str
    risk_level: str
    config_space: str
    feasibility_score: float
    novelty_score: float

    # Additional metadata
    source: str = "unknown"  # "grok", "dspy", "ai_scientist"
    generation_round: int = 0
    similar_works: list[str] = field(default_factory=list)
    refinement_history: list[str] = field(default_factory=list)

    @classmethod
    def from_optimization_idea(
        cls,
        idea: OptimizationIdea,
        source: str = "dspy",
    ) -> "Idea":
        """Convert from DSPy OptimizationIdea."""
        return cls(
            title=idea.title,
            description=idea.description,
            novelty=idea.novelty,
            implementation_sketch=idea.implementation_sketch,
            expected_outcome=idea.expected_outcome,
            risk_level=idea.risk_level,
            config_space=idea.config_space,
            feasibility_score=idea.feasibility_score,
            novelty_score=idea.novelty_score,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "novelty": self.novelty,
            "implementation_sketch": self.implementation_sketch,
            "expected_outcome": self.expected_outcome,
            "risk_level": self.risk_level,
            "config_space": self.config_space,
            "feasibility_score": self.feasibility_score,
            "novelty_score": self.novelty_score,
            "source": self.source,
            "generation_round": self.generation_round,
            "similar_works": self.similar_works,
        }

    @property
    def combined_score(self) -> float:
        """Combined score for ranking ideas."""
        # Weight novelty and feasibility equally
        return 0.5 * self.novelty_score + 0.5 * self.feasibility_score


class GrokIdeation:
    """
    Direct idea generation using Grok via OpenAI-compatible API.
    """

    SYSTEM_PROMPT = """You are an expert AI researcher specializing in PyTorch kernel optimization and GPU programming. Your task is to generate novel, implementable research ideas for PyTorch/Helion optimization.

For each idea, provide:
1. A clear, specific title
2. Detailed description of the approach
3. What makes it novel compared to existing work
4. High-level implementation sketch
5. Expected outcomes with specific metrics
6. Risk assessment
7. Config search space definition

Focus on ideas that:
- Can be implemented in 24 hours
- Are benchmarkable with clear metrics
- Target transformer-style workloads (attention, GEMM, softmax)
- Use PyTorch >= 2.4, torch.compile, or Helion DSL
- Involve config/hyperparameter search"""

    def __init__(self, config: LLMConfig):
        """
        Initialize Grok ideation.

        Args:
            config: LLM configuration
        """
        self.config = config

        # Initialize OpenAI client for Grok
        if config.provider == LLMProvider.GROK:
            self.client = OpenAI(
                api_key=config.grok_api_key,
                base_url=config.grok_base_url,
            )
            self.model = config.model
        else:
            # Use OpenAI-compatible interface for other providers
            self.client = OpenAI(
                api_key=config.active_api_key,
                base_url=config.active_base_url,
            )
            self.model = config.model

        logger.info(f"Initialized GrokIdeation with model {self.model}")

    def generate_ideas(
        self,
        literature_summary: str,
        constraints: list[str],
        num_ideas: int = 5,
    ) -> list[Idea]:
        """
        Generate ideas using Grok.

        Args:
            literature_summary: Summary of literature and gaps
            constraints: Implementation constraints
            num_ideas: Number of ideas to generate

        Returns:
            List of generated ideas
        """
        constraints_str = "\n".join(f"- {c}" for c in constraints)

        user_prompt = f"""Based on the following literature analysis and constraints, generate {num_ideas} novel PyTorch optimization ideas.

LITERATURE SUMMARY:
{literature_summary}

CONSTRAINTS:
{constraints_str}

For each idea, provide a JSON object with these fields:
- title: string
- description: string
- novelty: string (what makes this novel)
- implementation_sketch: string (how to implement in 24 hours)
- expected_outcome: string (specific expected improvements)
- risk_level: "low" | "medium" | "high"
- config_space: string (what parameters to search over)
- feasibility_score: float 0-1
- novelty_score: float 0-1

Return a JSON array of {num_ideas} idea objects."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content
            logger.debug(f"Grok response: {content[:500]}...")

            # Parse JSON response
            ideas_data = self._parse_ideas_json(content)

            ideas = []
            for data in ideas_data[:num_ideas]:
                try:
                    idea = Idea(
                        title=data.get("title", "Untitled"),
                        description=data.get("description", ""),
                        novelty=data.get("novelty", ""),
                        implementation_sketch=data.get("implementation_sketch", ""),
                        expected_outcome=data.get("expected_outcome", ""),
                        risk_level=data.get("risk_level", "medium"),
                        config_space=data.get("config_space", ""),
                        feasibility_score=float(data.get("feasibility_score", 0.5)),
                        novelty_score=float(data.get("novelty_score", 0.5)),
                        source="grok",
                    )
                    ideas.append(idea)
                except Exception as e:
                    logger.warning(f"Failed to parse idea: {e}")

            logger.info(f"Generated {len(ideas)} ideas from Grok")
            return ideas

        except Exception as e:
            logger.error(f"Grok generation failed: {e}")
            return []

    def _parse_ideas_json(self, content: str) -> list[dict[str, Any]]:
        """Parse JSON from LLM response."""
        # Try direct JSON parse
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return [data]
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in content
        import re

        array_match = re.search(r"\[[\s\S]*\]", content)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find JSON objects
        objects = re.findall(r"\{[^{}]+\}", content)
        results = []
        for obj_str in objects:
            try:
                results.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue

        return results


class AIScientistIdeation:
    """
    Ideation following AI Scientist patterns.

    Implements the workshop-style ideation flow from the AI Scientist framework.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        ideation_config: IdeationConfig,
    ):
        """
        Initialize AI Scientist ideation.

        Args:
            llm_config: LLM configuration
            ideation_config: Ideation-specific configuration
        """
        self.llm_config = llm_config
        self.ideation_config = ideation_config

        # Initialize DSPy
        configure_dspy_lm(llm_config)

        # DSPy modules
        self.idea_generator = IdeaGenerator()
        self.idea_refiner = IdeaRefiner()

        logger.info("Initialized AIScientistIdeation")

    def build_workshop_context(
        self,
        literature_summary: LiteratureSummary,
    ) -> str:
        """
        Build workshop context from literature summary.

        Mimics the AI Scientist's workshop file concept.

        Args:
            literature_summary: Literature analysis results

        Returns:
            Formatted workshop context string
        """
        context_parts = [
            "# PyTorch Kernel Optimization Workshop",
            "",
            "## Domain",
            f"{literature_summary.domain}",
            "",
            "## Key Findings from Literature",
        ]

        for paper in literature_summary.key_papers[:5]:
            context_parts.append(f"- {paper.title}: {paper.summary}")

        context_parts.extend([
            "",
            "## Identified Research Gaps",
        ])
        for gap in literature_summary.research_gaps[:5]:
            context_parts.append(
                f"- {gap.description} (Impact: {gap.potential_impact}, Difficulty: {gap.difficulty})"
            )

        context_parts.extend([
            "",
            "## Open Problems",
        ])
        for problem in literature_summary.open_problems[:5]:
            context_parts.append(f"- {problem}")

        context_parts.extend([
            "",
            "## Technical Constraints",
            "- PyTorch >= 2.4 with torch.compile",
            "- Helion DSL for custom kernels",
            "- Target: transformer workloads (attention, GEMM, softmax, layernorm)",
            "- Implementation time: 24 hours",
            "- Must be benchmarkable with clear metrics",
            "",
            "## Config Search Focus",
            "- Tiling and block sizes",
            "- Memory layout options",
            "- Algorithm variants",
            "- torch.compile modes (default, reduce-overhead, max-autotune)",
        ])

        if literature_summary.x_threads:
            context_parts.extend([
                "",
                "## Recent X Threads (full text)",
            ])
            for thread in literature_summary.x_threads:
                context_parts.append(thread.to_summary_string())

        return "\n".join(context_parts)

    def generate_ideas(
        self,
        literature_summary: LiteratureSummary,
    ) -> list[Idea]:
        """
        Generate ideas using AI Scientist patterns.

        Follows a multi-round generation and refinement process.

        Args:
            literature_summary: Literature analysis results

        Returns:
            List of generated and refined ideas
        """
        # Build workshop context
        workshop_context = self.build_workshop_context(literature_summary)
        constraints_str = "\n".join(self.ideation_config.constraints)

        all_ideas: list[Idea] = []

        # Multi-round generation
        for round_num in range(self.ideation_config.num_generation_rounds):
            logger.info(f"Generation round {round_num + 1}/{self.ideation_config.num_generation_rounds}")

            # Generate ideas via DSPy
            optimization_ideas = self.idea_generator(
                literature_summary=workshop_context,
                constraints=constraints_str,
                num_ideas=self.ideation_config.ideas_per_round,
            )

            # Convert to Idea objects
            for opt_idea in optimization_ideas:
                idea = Idea.from_optimization_idea(opt_idea, source="ai_scientist")
                idea.generation_round = round_num + 1
                all_ideas.append(idea)

            logger.info(f"Round {round_num + 1}: Generated {len(optimization_ideas)} ideas")

        # Refine top ideas
        if all_ideas:
            all_ideas = self._refine_ideas(all_ideas, constraints_str)

        return all_ideas

    def _refine_ideas(
        self,
        ideas: list[Idea],
        constraints: str,
    ) -> list[Idea]:
        """Refine ideas based on feasibility feedback."""
        refined_ideas: list[Idea] = []

        # Sort by initial scores
        sorted_ideas = sorted(ideas, key=lambda x: x.combined_score, reverse=True)

        for idea in sorted_ideas[:self.ideation_config.max_ideas]:
            # Generate feedback
            feedback = f"""
            Please improve this idea for better feasibility:
            - Ensure the config space is clearly defined with concrete parameters
            - Make the implementation sketch more specific
            - Clarify the expected metrics and success criteria
            - Reduce scope if risk_level is high
            """

            try:
                opt_idea = OptimizationIdea(
                    title=idea.title,
                    description=idea.description,
                    novelty=idea.novelty,
                    implementation_sketch=idea.implementation_sketch,
                    expected_outcome=idea.expected_outcome,
                    risk_level=idea.risk_level,
                    config_space=idea.config_space,
                    feasibility_score=idea.feasibility_score,
                    novelty_score=idea.novelty_score,
                )

                refined_opt = self.idea_refiner(
                    original_idea=opt_idea,
                    feedback=feedback,
                    constraints=constraints,
                )

                refined = Idea.from_optimization_idea(refined_opt, source="ai_scientist")
                refined.generation_round = idea.generation_round
                refined.refinement_history = [f"Refined from: {idea.title}"]
                refined_ideas.append(refined)

            except Exception as e:
                logger.warning(f"Refinement failed for '{idea.title}': {e}")
                refined_ideas.append(idea)

        return refined_ideas


class NoveltyChecker:
    """
    Check novelty of ideas against existing literature.
    """

    def __init__(
        self,
        exa_config: ExaConfig,
        llm_config: LLMConfig,
    ):
        """
        Initialize novelty checker.

        Args:
            exa_config: Exa API configuration
            llm_config: LLM configuration
        """
        self.literature = LiteratureDiscovery(exa_config, llm_config)
        self.llm_config = llm_config

        # Configure DSPy
        configure_dspy_lm(llm_config)
        self.novelty_assessor = NoveltyAssessor()

    def check_novelty(
        self,
        idea: Idea,
        search_limit: int = 5,
    ) -> tuple[float, list[str]]:
        """
        Check novelty of an idea.

        Args:
            idea: The idea to check
            search_limit: Number of similar papers to search for

        Returns:
            Tuple of (novelty_score, list of similar works)
        """
        # Search for similar work
        query = f"{idea.title} {idea.description[:200]}"
        results = self.literature.search_papers(query, num_results=search_limit)

        if not results:
            logger.info(f"No similar papers found for '{idea.title}'")
            return 0.9, []  # High novelty if nothing found

        # Format similar papers
        similar_papers = "\n".join([
            f"- {r.title}: {r.text[:300] if r.text else 'No summary'}"
            for r in results
        ])

        # Use DSPy to assess novelty
        assessment = self.novelty_assessor(
            idea_title=idea.title,
            idea_description=idea.description,
            similar_papers=similar_papers,
        )

        logger.info(
            f"Novelty check for '{idea.title}': "
            f"score={assessment.novelty_score:.2f}, "
            f"similar_works={len(assessment.similar_works)}"
        )

        return assessment.novelty_score, assessment.similar_works


class IdeationPipeline:
    """
    Complete ideation pipeline combining multiple approaches.
    """

    def __init__(self, config: ResearchConfig):
        """
        Initialize ideation pipeline.

        Args:
            config: Research configuration
        """
        self.config = config

        # Initialize components
        self.grok_ideation = GrokIdeation(config.llm)
        self.ai_scientist = AIScientistIdeation(config.llm, config.ideation)
        self.novelty_checker = NoveltyChecker(config.exa, config.llm)

        logger.info("Initialized IdeationPipeline")

    def generate_ideas(
        self,
        literature_summary: LiteratureSummary,
        check_novelty: bool = True,
    ) -> list[Idea]:
        """
        Generate ideas using multiple approaches.

        Args:
            literature_summary: Literature analysis results
            check_novelty: Whether to check novelty via Exa

        Returns:
            Sorted list of ideas
        """
        all_ideas: list[Idea] = []

        # Generate from AI Scientist approach
        logger.info("Generating ideas via AI Scientist approach...")
        ai_ideas = self.ai_scientist.generate_ideas(literature_summary)
        all_ideas.extend(ai_ideas)

        # Generate from Grok
        logger.info("Generating ideas via Grok...")
        grok_ideas = self.grok_ideation.generate_ideas(
            literature_summary=literature_summary.to_summary_string(),
            constraints=self.config.ideation.constraints,
            num_ideas=self.config.ideation.ideas_per_round,
        )
        all_ideas.extend(grok_ideas)

        # Deduplicate by title similarity
        all_ideas = self._deduplicate_ideas(all_ideas)

        # Check novelty if requested
        if check_novelty:
            logger.info("Checking novelty of ideas...")
            for idea in all_ideas:
                try:
                    novelty_score, similar = self.novelty_checker.check_novelty(
                        idea,
                        self.config.ideation.novelty_search_limit,
                    )
                    idea.novelty_score = novelty_score
                    idea.similar_works = similar
                except Exception as e:
                    logger.warning(f"Novelty check failed for '{idea.title}': {e}")

        # Filter and sort
        filtered_ideas = [
            idea for idea in all_ideas
            if idea.novelty_score >= self.config.ideation.novelty_threshold
            and idea.feasibility_score >= self.config.ideation.feasibility_threshold
        ]

        # Sort by combined score
        sorted_ideas = sorted(
            filtered_ideas,
            key=lambda x: x.combined_score,
            reverse=True,
        )

        # Limit to max ideas
        final_ideas = sorted_ideas[:self.config.ideation.max_ideas]

        logger.info(
            f"Ideation complete: {len(all_ideas)} generated, "
            f"{len(filtered_ideas)} passed filters, "
            f"{len(final_ideas)} selected"
        )

        return final_ideas

    def _deduplicate_ideas(self, ideas: list[Idea]) -> list[Idea]:
        """Remove duplicate ideas based on title similarity."""
        unique_ideas: list[Idea] = []
        seen_titles: set[str] = set()

        for idea in ideas:
            # Simple deduplication by normalized title
            normalized = idea.title.lower().strip()
            if normalized not in seen_titles:
                unique_ideas.append(idea)
                seen_titles.add(normalized)

        return unique_ideas


def generate_ideas(
    literature_summary: LiteratureSummary,
    config: ResearchConfig | None = None,
    check_novelty: bool = True,
) -> list[Idea]:
    """
    Convenience function for idea generation.

    Args:
        literature_summary: Literature analysis results
        config: Optional research configuration
        check_novelty: Whether to check novelty

    Returns:
        Sorted list of ideas
    """
    if config is None:
        config = ResearchConfig()

    pipeline = IdeationPipeline(config)
    return pipeline.generate_ideas(literature_summary, check_novelty)
