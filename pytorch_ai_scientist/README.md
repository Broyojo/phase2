# PyTorch Scientist

**AI Scientist for PyTorch Optimization** - An automated research system that discovers, generates, and validates optimization ideas for PyTorch kernels.

## Overview

PyTorch Scientist combines multiple AI systems to automate the research and development cycle for PyTorch kernel optimization:

1. **Literature Discovery** (Exa.ai) - Searches academic papers to identify research gaps
2. **Idea Generation** (AI Scientist + Grok + DSPy) - Generates novel, implementable optimization ideas
3. **Config Search** (OpenEvolve patterns) - Evolutionary/MCTS search over configuration spaces
4. **Experiment Execution** (PyTorch/Helion) - Benchmarks configurations and measures speedups
5. **Summary Generation** (DSPy) - Produces human-readable reports of findings

## Features

- **DSPy-Optimized Prompts**: All LLM interactions use DSPy signatures and modules for prompt optimization
- **Multiple LLM Providers**: Supports Grok, Anthropic Claude, and OpenAI
- **Structured Config Spaces**: Helion kernel configs, torch.compile options, softmax/attention parameters
- **Evolutionary Search**: GA-based search with tournament selection, crossover, and mutation
- **MCTS Support**: Monte Carlo Tree Search for structured exploration
- **Comprehensive Benchmarking**: GPU timing with CUDA events, warmup handling, statistical analysis

## Installation

### Requirements

- Python 3.10+
- PyTorch >= 2.4.0 (optional, for experiments)
- CUDA-capable GPU (optional, for GPU experiments)

### Install from source

```bash
git clone https://github.com/your-org/pytorch-scientist.git
cd pytorch-scientist
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

### Install with Helion/Triton support

```bash
pip install -e ".[helion]"
```

## Configuration

Set the following environment variables:

```bash
# Required for literature search
export EXA_API_KEY="your-exa-api-key"

# Optional: X posts enrichment
export X_API_KEY="your-x-api-key"

# Required for LLM calls (choose one)
export XAI_API_KEY="your-grok-api-key"       # For Grok
export ANTHROPIC_API_KEY="your-anthropic-key" # For Claude
export OPENAI_API_KEY="your-openai-key"       # For OpenAI
```

## Quick Start

### Run a Quick Demo

```bash
# Run with minimal settings
python -m pytorch_scientist.cli demo

# Or use the script
python scripts/run_full_pipeline.py --max-ideas 3 --max-evals 20
```

### Run Full Pipeline

```bash
# Full pipeline with defaults
python scripts/run_full_pipeline.py

# Custom domain and parameters
python scripts/run_full_pipeline.py \
    --domain "PyTorch attention optimization" \
    --max-ideas 5 \
    --max-evals 100 \
    --operation attention \
    --llm-provider grok
```

### Discovery Only (No Experiments)

```bash
python scripts/run_discovery.py --domain "transformer kernel optimization"
```

## Usage

### Python API

```python
from pytorch_scientist import ResearchConfig, run_pytorch_scientist

# Configure the run
config = ResearchConfig(
    domain="LLM guided PyTorch kernel optimization",
)
config.ideation.max_ideas = 5
config.search.max_evaluations = 50

# Run the full pipeline
result = run_pytorch_scientist(config)

print(f"Best speedup: {result['best_speedup']:.4f}x")
print(f"Results saved to: {result['run_dir']}")
```

### CLI

```bash
# Full pipeline
pytorch-scientist run --domain "PyTorch optimization" --max-evals 100

# Discovery only
pytorch-scientist discover --max-ideas 10

# Quick demo
pytorch-scientist demo
```

## Project Structure

```
pytorch_scientist/
├── __init__.py           # Package exports
├── config.py             # Configuration dataclasses
├── cli.py                # Command-line interface
├── literature.py         # Exa.ai integration, gap analysis
├── ideation.py           # AI Scientist + Grok + DSPy ideation
├── search_space.py       # Config space definitions
├── experiments.py        # Benchmark harness
├── search_engine.py      # OpenEvolve-style search
├── pipeline.py           # End-to-end orchestration
├── dspy_programs.py      # DSPy signatures and modules
└── utils/
    ├── logging.py        # Logging configuration
    ├── persistence.py    # JSON/YAML helpers
    └── timing.py         # Benchmark timing utilities

scripts/
├── run_full_pipeline.py  # Full pipeline script
└── run_discovery.py      # Discovery-only script

tests/
├── test_config.py
├── test_literature.py
├── test_search_space.py
├── test_search_engine.py
├── test_ideation.py
└── test_utils.py
```

## Configuration Options

### LLM Configuration

```python
from pytorch_scientist.config import LLMConfig, LLMProvider

llm_config = LLMConfig(
    provider=LLMProvider.GROK,      # GROK, ANTHROPIC, or OPENAI
    model="grok-3-mini",            # Model name
    temperature=0.7,
    max_tokens=4096,
)
```

### Search Configuration

```python
from pytorch_scientist.config import SearchConfig, SearchStrategy

search_config = SearchConfig(
    strategy=SearchStrategy.EVOLUTIONARY,  # EVOLUTIONARY, MCTS, or RANDOM
    max_evaluations=50,
    population_size=10,
    mutation_rate=0.2,
    crossover_rate=0.7,
    elite_count=2,
)
```

### Experiment Configuration

```python
from pytorch_scientist.config import ExperimentConfig, TargetOperation

experiment_config = ExperimentConfig(
    target_operation=TargetOperation.SOFTMAX,  # SOFTMAX, GEMM, or ATTENTION
    device="cuda",
    warmup_iterations=10,
    benchmark_iterations=100,
)
```

## Output Artifacts

Each run creates a timestamped directory in `runs/` with:

```
runs/run_20241205_120000/
├── metadata.json           # Run configuration and results
├── literature/
│   └── literature_summary.json
├── ideas/
│   ├── ideas.json          # All generated ideas
│   └── selected_idea.json  # The selected idea
├── search/
│   └── search_history.json # Config search trajectory
├── results/
│   ├── best_result.json    # Best configuration found
│   └── summary.md          # Human-readable report
└── run.log                 # Detailed logs
```

## DSPy Integration

All LLM interactions are structured as DSPy signatures:

```python
from pytorch_scientist.dspy_programs import (
    LiteratureSummarizer,
    IdeaGenerator,
    NoveltyAssessor,
    ExperimentSummarizer,
)

# Configure DSPy with your LLM
from pytorch_scientist.dspy_programs import configure_dspy_lm
configure_dspy_lm(llm_config)

# Use DSPy modules
summarizer = LiteratureSummarizer()
result = summarizer(
    topic="PyTorch kernel optimization",
    paper_summaries=paper_text,
    constraints="Focus on transformers",
)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=pytorch_scientist

# Run specific test file
pytest tests/test_search_space.py -v
```

## Architecture

### Phase 1: Literature Discovery

1. Search Exa.ai for relevant papers
2. Query SOTA methods with predefined queries
3. Use DSPy `LiteratureSummarizer` to extract gaps and trends

### Phase 2: Idea Generation

1. Build workshop context from literature
2. Generate ideas via DSPy `IdeaGenerator`
3. Optional: Generate additional ideas via Grok direct API
4. Check novelty via Exa search + DSPy `NoveltyAssessor`
5. Rank and select top idea

### Phase 3: Config Search

1. Define config space (Helion, torch.compile, softmax, etc.)
2. Initialize population with random configs
3. Evolutionary loop:
   - Evaluate configs via benchmark
   - Select, crossover, mutate
   - Track best config
4. Return search result with history

### Phase 4: Summary Generation

1. Use DSPy `ExperimentSummarizer` to generate report
2. Save artifacts (JSON, Markdown)
3. Return results to user

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project integrates concepts and code from:

- [AI Scientist](https://github.com/SakanaAI/AI-Scientist) - Automated research framework
- [DSPy](https://github.com/stanfordnlp/dspy) - Prompt optimization
- [Exa.ai](https://exa.ai/) - Neural search for papers
- [PyTorch Helion](https://github.com/pytorch/helion) - Kernel DSL
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) - Evolutionary search
