"""
Plotting module for causal-audit package.

Single entry point: generate_all_figures() produces all 5 publication-quality
figures in IEEE style (SciencePlots).
"""

from .figures import generate_all_figures

__all__ = [
    "generate_all_figures",
]
