"""
Plotting modules for causal-audit package.

v0.1: 4 critical figures (risk posteriors, decision, time series, correlation)
v0.2: Full 12-figure gallery (data overview, diagnostics, pairwise, etc.)
"""

from .core_plots_v01 import (
    plot_risk_posteriors,
    plot_decision_summary,
    plot_time_series_overview,
    plot_correlation_heatmap
)

__all__ = [
    "plot_risk_posteriors",
    "plot_decision_summary", 
    "plot_time_series_overview",
    "plot_correlation_heatmap"
]

