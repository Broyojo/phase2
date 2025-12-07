"""
PyTorch Scientist - AI Scientist for PyTorch Optimization

An automated research system that combines:
- Exa.ai for literature discovery
- DSPy for prompt optimization
- Grok/LLMs for ideation and analysis
- OpenEvolve for config search
- Helion/PyTorch for kernel experimentation
"""

__version__ = "0.1.0"

from pytorch_scientist.config import ResearchConfig, LLMConfig
from pytorch_scientist.pipeline import run_pytorch_scientist

__all__ = [
    "ResearchConfig",
    "LLMConfig",
    "run_pytorch_scientist",
    "__version__",
]
