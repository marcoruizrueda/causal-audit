"""
Core Plotting Functions for causal-audit v0.1

Implements 4 CRITICAL figures:
1. Risk posteriors (with 95% CrI)
2. Decision summary (strategy probabilities or abstention explanation)
3. Time series overview (all variables)
4. Correlation heatmap

v0.2 will add 8 more figures (diagnostics, pairwise, etc.)
"""

from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def plot_risk_posteriors(
    risk_profile: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    Plot 1: Risk Posteriors with 95% CrI (violin plots).
    
    Args:
        risk_profile: RiskProfile dictionary
        output_path: Where to save figure
        
    Returns:
        Path to saved figure
    """
    risks = risk_profile["risks"]
    risk_names = list(risks.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    positions = np.arange(len(risk_names))
    means = [risks[r]["mean"] for r in risk_names]
    lowers = [risks[r]["lower_95"] for r in risk_names]
    uppers = [risks[r]["upper_95"] for r in risk_names]
    
    # Plot error bars (95% CrI)
    for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
        # Color by risk level
        if mean < 0.3:
            color = 'green'
        elif mean < 0.7:
            color = 'orange'
        else:
            color = 'red'
        
        # Plot CrI as error bar
        ax.errorbar(i, mean, yerr=[[mean - lower], [upper - mean]],
                    fmt='o', markersize=10, capsize=8, capthick=2,
                    color=color, ecolor=color, alpha=0.7, linewidth=2)
    
    # Horizontal reference lines
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels([r.replace("Risk", "\nRisk") for r in risk_names], rotation=0)
    ax.set_ylabel("Risk Posterior (0-1)")
    ax.set_title("Composite Risk Scores with 95% Credible Intervals", fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # Add threshold labels
    ax.text(len(risk_names) - 0.5, 0.3, "Low", va='bottom', ha='right', fontsize=8, color='gray')
    ax.text(len(risk_names) - 0.5, 0.7, "High", va='bottom', ha='right', fontsize=8, color='gray')
    
    plt.tight_layout()
    
    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "displayed"


def plot_decision_summary(
    policy: Dict[str, Any],
    risk_profile: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    Plot 2: Decision Summary (strategy probabilities or abstention).
    
    Args:
        policy: Policy dictionary
        risk_profile: RiskProfile dictionary
        output_path: Where to save figure
        
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    decision = policy["decision"]
    
    if decision == "abstain":
        # Plot abstention explanation
        risks = risk_profile["risks"]
        risk_names = list(risks.keys())
        
        # Sort by mean (descending)
        sorted_risks = sorted(
            [(name, risks[name]["mean"], risks[name]["upper_95"] - risks[name]["lower_95"]) 
             for name in risk_names],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Bar chart of top risks
        names = [r[0].replace("Risk", "") for r in sorted_risks]
        means = [r[1] for r in sorted_risks]
        widths = [r[2] for r in sorted_risks]
        
        bars = ax.barh(names, means, color='red', alpha=0.6)
        
        # Add CrI width as error bars
        ax.errorbar(means, range(len(names)), xerr=np.array(widths)/2,
                    fmt='none', ecolor='darkred', capsize=5, linewidth=2)
        
        ax.set_xlabel("Risk Level")
        ax.set_title("ABSTENTION: Top Risk Contributors", fontweight='bold', color='red')
        ax.set_xlim(0, 1.0)
        
        # Add text
        reason_codes = policy.get("policy_reason_codes", [])
        reason_text = "\n".join([f"• {r.replace('_', ' ')}" for r in reason_codes[:3]])
        ax.text(0.98, 0.02, f"Reasons:\n{reason_text}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    else:
        # Plot recommendation (strategy probabilities would go here)
        # For v0.1, just show the decision
        method = policy.get("recommended_method", "Unknown")
        confidence = policy.get("confidence", 0.0)
        
        ax.text(0.5, 0.5, f"RECOMMENDED:\n{method}\n\nConfidence: {confidence:.1%}",
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=1.0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Recommendation", fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "displayed"


def plot_time_series_overview(
    data: pd.DataFrame,
    output_path: Optional[str] = None
) -> str:
    """
    Plot 3: Time Series Overview (all variables).
    
    Args:
        data: Time-series DataFrame
        output_path: Where to save figure
        
    Returns:
        Path to saved figure
    """
    n_vars = len(data.columns)
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2 * n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    for i, col in enumerate(data.columns):
        ax = axes[i]
        
        # Plot time series
        ax.plot(data.index, data[col], linewidth=1, alpha=0.8, color='steelblue')
        
        # Add mean line
        mean_val = data[col].mean()
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Formatting
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
        
        # Add stats text
        missing_pct = data[col].isna().mean() * 100
        stats_text = f"μ={mean_val:.2f}, σ={data[col].std():.2f}, Missing={missing_pct:.1f}%"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    axes[-1].set_xlabel("Time")
    fig.suptitle("Time Series Overview", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "displayed"


def plot_correlation_heatmap(
    data: pd.DataFrame,
    output_path: Optional[str] = None
) -> str:
    """
    Plot 4: Correlation Heatmap.
    
    Args:
        data: Time-series DataFrame
        output_path: Where to save figure
        
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Compute correlation matrix
    corr = data.corr()
    
    # Plot heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax
    )
    
    ax.set_title("Cross-Correlation Matrix (Lag 0)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "displayed"


# ============================================================================
# v0.2 STUBS (to be implemented)
# ============================================================================

def plot_full_diagnostics(audit_evidence: Dict[str, Any], output_dir: str):
    """
    v0.2: Full diagnostic plots (8 figures).
    
    Will include:
    - Stationarity diagnostics (ADF/KPSS + breaks)
    - Seasonality periodograms
    - Nonlinearity tests
    - Gaussianity Q-Q plots
    - Persistence (Hurst + T_eff)
    - Confounding (Chow tests)
    - Irregularity (gap analysis)
    - Transport/upwind (EO plugin)
    """
    raise NotImplementedError("Full diagnostics plotting deferred to v0.2")


def plot_pairwise_analysis(data: pd.DataFrame, output_dir: str):
    """
    v0.2: Pairwise relationship plots (3 figures).
    
    Will include:
    - Scatter matrix
    - Lagged cross-correlation functions
    - Granger preview heatmap
    """
    raise NotImplementedError("Pairwise plotting deferred to v0.2")


def plot_extended_recommendation(
    policy: Dict[str, Any],
    risk_profile: Dict[str, Any],
    output_dir: str
):
    """
    v0.2: Extended recommendation plots (3 figures).
    
    Will include:
    - Method suitability heatmap
    - Decision flowchart
    - Compute savings estimate
    """
    raise NotImplementedError("Extended recommendation plotting deferred to v0.2")

