"""
Plotting module for causal-audit package.

Single entry point: generate_all_figures() produces six diagnostic figures
in IEEE style (SciencePlots), with an optional exploratory discovery graph.
"""

from .figures import generate_all_figures

__all__ = [
    "generate_all_figures",
]
