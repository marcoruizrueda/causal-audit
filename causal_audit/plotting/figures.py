"""
Publication-quality pre-discovery figures for causal-audit.

Single consolidated plotting module. Generates all figures in IEEE style
(SciencePlots). All outputs at 300 DPI in both PDF and PNG.

Generated figures:
  1. prediscovery_summary.pdf/.png   -- Main composite (panels a-e)
  2. correlation_and_lagged_crosscorrelation.pdf/.png
  3. spectral_density.pdf/.png
  4. stationarity_diagnostic.pdf/.png
  5. sample_size_adequacy.pdf/.png
  6. assumption_deep_dive.pdf/.png   -- Per-assumption diagnostic + action guide

Install requirement: pip install SciencePlots

Style options (change DEFAULT_STYLE):
  "ieee"    -- IEEE Transactions (default, recommended for journal papers)
  "science" -- Generic scientific style
  "nature"  -- Nature-family journals
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import warnings

import numpy as np
import pandas as pd

# Default style -- change this line to switch styles globally
# Options: "ieee", "science", "nature"
DEFAULT_STYLE = "ieee"


def generate_all_figures(
    data: pd.DataFrame,
    audit_evidence_dict: Dict[str, Any],
    risk_profile_dict: Dict[str, Any],
    policy_dict: Dict[str, Any],
    output_dir: Path,
    style: str = DEFAULT_STYLE,
    tau_max: int = 20,
) -> None:
    """
    Generate all pre-discovery figures.

    Parameters
    ----------
    data : pd.DataFrame
        Raw time series data (rows=time, columns=variables).
    audit_evidence_dict : dict
        From audit_evidence.to_dict().
    risk_profile_dict : dict
        From risk_profile.to_dict().
    policy_dict : dict
        From policy.to_dict().
    output_dir : Path
        Output directory (figures/ subfolder will be created).
    style : str
        SciencePlots style. Options: "ieee", "science", "nature".
    tau_max : int
        Maximum lag for cross-correlation computation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_list = _resolve_style(style)

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cols = data.select_dtypes(include=["number"]).columns.tolist()

    def _ctx():
        if style_list:
            return plt.style.context(style_list)
        from contextlib import nullcontext

        return nullcontext()

    def _save(fig, name):
        fig.savefig(figures_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    with _ctx():
        _plot_prediscovery_summary(
            data,
            cols,
            audit_evidence_dict,
            risk_profile_dict,
            policy_dict,
            figures_dir,
            _save,
        )

    with _ctx():
        _plot_correlation_and_lagged(data, cols, tau_max, _save)

    with _ctx():
        _plot_spectral_density(data, cols, _save)

    with _ctx():
        _plot_stationarity_diagnostic(data, cols, audit_evidence_dict, _save)

    with _ctx():
        _plot_sample_size_adequacy(data, cols, audit_evidence_dict, _save)

    with _ctx():
        _plot_assumption_deep_dive(
            data, cols, audit_evidence_dict, risk_profile_dict, _save
        )

    with _ctx():
        _plot_dependency_network(data, cols, audit_evidence_dict, _save)


def _resolve_style(style: str):
    """Resolve SciencePlots style list, returning None if unavailable."""
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        warnings.warn(
            "SciencePlots not installed (pip install SciencePlots). "
            "Falling back to default matplotlib style."
        )
        return None

    if style == "ieee":
        return ["science", "ieee", "no-latex"]
    elif style == "nature":
        return ["science", "nature", "no-latex"]
    else:
        return ["science", "no-latex"]


# ==============================================================================
# Figure 1: Pre-discovery summary (composite, panels a-e)
# ==============================================================================


def _plot_prediscovery_summary(
    data, cols, audit_evidence_dict, risk_profile_dict, policy_dict, figures_dir, _save
):
    """Main composite figure with 5 panels. Tight layout, clear labels."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    try:
        from statsmodels.tsa.stattools import acf as compute_acf
    except ImportError:
        compute_acf = None

    risks = risk_profile_dict.get("risks", {})
    rec = policy_dict.get("recommended_method", "N/A")
    conf = policy_dict.get("confidence", 0) or 0
    n_vars = len(cols)

    RISK_NAMES = [
        "Nonstationarity",
        "Nonlinearity",
        "Confounding",
        "Irregularity",
        "Persistence",
        "Seasonality",
    ]
    RISK_KEYS = [
        "NonstationarityRisk",
        "NonlinearityRisk",
        "ConfoundingRisk",
        "IrregularityRisk",
        "PersistenceRisk",
        "SeasonalityRisk",
    ]
    vals = [risks.get(k, {}).get("mean", 0) for k in RISK_KEYS]
    err_lo = [
        max(0, v - risks.get(k, {}).get("lower_95", 0)) for v, k in zip(vals, RISK_KEYS)
    ]
    err_hi = [
        max(0, risks.get(k, {}).get("upper_95", 0) - v) for v, k in zip(vals, RISK_KEYS)
    ]

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.38,
        wspace=0.35,
        height_ratios=[1.8, 1.4, 1.4],
    )

    cmap = plt.cm.Set2(np.linspace(0, 1, max(n_vars, 1)))

    # -- (a) Time series overview --
    ax_ts = fig.add_subplot(gs[0, :])
    for i, col in enumerate(cols):
        s = data[col].dropna()
        s_norm = (s - s.mean()) / (s.std() + 1e-10)
        ax_ts.plot(
            s_norm.values, color=cmap[i], lw=0.6, alpha=0.85, label=col, rasterized=True
        )
    ax_ts.set_xlabel("Time step", fontsize=9)
    ax_ts.set_ylabel("Standardized value", fontsize=9)
    ax_ts.tick_params(labelsize=8)
    ax_ts.legend(
        ncol=min(n_vars, 5),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        fontsize=7,
        columnspacing=1.0,
    )
    # Annotate dataset dimensions
    T_len = len(data)
    ax_ts.text(
        0.99,
        0.95,
        f"T={T_len}, N={n_vars}",
        transform=ax_ts.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.3, alpha=0.9),
    )
    ax_ts.text(
        -0.02,
        1.08,
        "(a)",
        transform=ax_ts.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )

    # -- (b) Risk profile --
    ax_rp = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(RISK_NAMES))
    bar_colors = [
        "#4DAF4A" if v < 0.30 else "#FF7F00" if v < 0.60 else "#E41A1C" for v in vals
    ]
    ax_rp.barh(
        y_pos,
        vals,
        xerr=[err_lo, err_hi],
        color=bar_colors,
        height=0.55,
        edgecolor="white",
        linewidth=0.4,
        error_kw=dict(elinewidth=0.7, capsize=2.5, capthick=0.7),
    )
    ax_rp.axvline(0.30, color="#FF7F00", lw=0.7, ls="--", alpha=0.5)
    ax_rp.axvline(0.60, color="#E41A1C", lw=0.7, ls="--", alpha=0.5)
    ax_rp.set_yticks(y_pos)
    ax_rp.set_yticklabels(RISK_NAMES, fontsize=7.5)
    ax_rp.set_xlabel("Risk score", fontsize=9)
    ax_rp.set_xlim(0, 1.15)
    ax_rp.tick_params(labelsize=8)
    # Value labels: offset slightly from error bar cap
    for i, v in enumerate(vals):
        upper_end = v + err_hi[i]
        label_x = min(upper_end + 0.04, 1.08)
        ax_rp.text(label_x, i, f"{v:.2f}", va="center", fontsize=7, color="#333333")
    ax_rp.text(
        -0.02,
        1.06,
        "(b)",
        transform=ax_rp.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )

    # -- (c) ACF overlay with T_eff annotation --
    ax_acf = fig.add_subplot(gs[1, 1])
    # Use high-contrast colors for ACF lines (tab10 is more distinguishable)
    acf_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    if compute_acf is not None:
        conf_band = 1.96 / np.sqrt(max(len(data), 1))
        n_show = min(n_vars, 4)
        for i, col in enumerate(cols[:n_show]):
            s = data[col].dropna().values
            lags = min(25, len(s) // 4)
            if len(s) > 20:
                try:
                    av = compute_acf(s, nlags=lags, fft=True)
                    ax_acf.plot(
                        range(1, len(av)),
                        av[1:],
                        color=acf_colors[i % len(acf_colors)],
                        lw=1.1,
                        alpha=0.9,
                        label=col,
                    )
                except Exception:
                    pass
        ax_acf.axhline(conf_band, color="gray", lw=0.5, ls="--")
        ax_acf.axhline(-conf_band, color="gray", lw=0.5, ls="--")
        ax_acf.axhline(0, color="black", lw=0.3)
        # Annotate mean T_eff
        persistence = audit_evidence_dict.get("diagnostics", {}).get("persistence", {})
        t_effs = [persistence.get(c, {}).get("t_eff", len(data)) for c in cols]
        mean_teff = np.mean(t_effs)
        ax_acf.text(
            0.97,
            0.95,
            f"Mean $T_{{eff}}$={mean_teff:.0f}",
            transform=ax_acf.transAxes,
            fontsize=7,
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.4, alpha=0.9
            ),
        )
    ax_acf.set_xlabel("Lag", fontsize=9)
    ax_acf.set_ylabel("ACF", fontsize=9)
    ax_acf.tick_params(labelsize=8)
    # Legend with background for readability over lines near zero
    ax_acf.legend(
        frameon=True,
        fontsize=7,
        ncol=2,
        loc="upper right",
        fancybox=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )
    ax_acf.text(
        -0.02,
        1.06,
        "(c)",
        transform=ax_acf.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )

    # -- (d) Nonlinearity scatter with diagnostic ratio --
    ax_nl = fig.add_subplot(gs[2, 0])
    pnl = audit_evidence_dict.get("diagnostics", {}).get("pairwise_nonlinearity", {})
    pairs = pnl.get("pairs", {})
    r2s = [v.get("r_squared", 0) for v in pairs.values()]
    mis = [v.get("mi_estimate", 0) for v in pairs.values()]
    nls = [str(v.get("is_nonlinear", False)).lower() == "true" for v in pairs.values()]
    fn = pnl.get("summary", {}).get("fraction_nonlinear", 0)
    ci_rec = pnl.get("summary", {}).get("recommended_ci_test", "parcorr")

    if r2s:
        cs = ["#E41A1C" if nl else "#377EB8" for nl in nls]
        ax_nl.scatter(r2s, mis, c=cs, s=18, alpha=0.7, edgecolors="none")
        lim = max(max(r2s, default=0.1), max(mis, default=0.1)) * 1.15
        ax_nl.plot([0, lim], [0, lim], "k--", lw=0.5, alpha=0.4)
        ax_nl.set_xlim(0, lim)
        ax_nl.set_ylim(0, lim)
    ax_nl.set_xlabel(r"Pearson $r^2$", fontsize=9)
    ax_nl.set_ylabel("MI (bits)", fontsize=9)
    ax_nl.tick_params(labelsize=8)
    ax_nl.set_title(f"{fn:.0%} nonlinear | CI: {ci_rec.upper()}", fontsize=8, pad=4)
    ax_nl.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#E41A1C",
                markersize=5,
                label="Nonlinear",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#377EB8",
                markersize=5,
                label="Linear",
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )
    ax_nl.text(
        -0.02,
        1.06,
        "(d)",
        transform=ax_nl.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )

    # -- (e) Recommendation summary --
    ax_rec = fig.add_subplot(gs[2, 1])
    ax_rec.axis("off")
    text_lines = [
        f"Recommendation: {rec}",
        f"Confidence: {conf:.0%}",
        "",
        "Key findings:",
    ]
    for name, val in zip(RISK_NAMES, vals):
        if val >= 0.60:
            text_lines.append(f"  {name}: HIGH ({val:.2f})")
        elif val >= 0.30:
            text_lines.append(f"  {name}: moderate ({val:.2f})")

    if not any(v >= 0.30 for v in vals):
        text_lines.append("  All risks low")
        text_lines.append("  Proceed with standard analysis")

    text_lines.append("")
    text_lines.append("Suggested action:")
    if vals[0] >= 0.60:
        text_lines.append("  Detrend/deseasonalize before discovery")
    if vals[1] >= 0.35:
        text_lines.append("  Use CMIknn (nonlinear CI test)")
    if vals[2] >= 0.45:
        text_lines.append("  Use LPCMCI (latent confounders)")
    if all(v < 0.30 for v in vals):
        text_lines.append("  Use PCMCI+ with ParCorr")

    ax_rec.text(
        0.05,
        0.95,
        "\n".join(text_lines),
        transform=ax_rec.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F8F8F8",
            edgecolor="#CCCCCC",
            linewidth=0.6,
        ),
    )
    ax_rec.text(
        -0.02,
        1.06,
        "(e)",
        transform=ax_rec.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )

    # Save
    fig.savefig(figures_dir / "prediscovery_summary.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "prediscovery_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Figure 2: Correlation & Lagged Cross-Correlation
# ==============================================================================


def _plot_correlation_and_lagged(data, cols, tau_max, _save):
    """Side-by-side contemporaneous and lagged cross-correlation matrices."""
    import matplotlib.pyplot as plt

    n_vars = len(cols)
    fig, (ax_corr, ax_lag) = plt.subplots(1, 2, figsize=(7.0, 3.8))

    # Left: Pearson correlation heatmap
    corr_matrix = data[cols].corr().values
    im1 = ax_corr.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_corr.set_xticks(range(n_vars))
    ax_corr.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax_corr.set_yticks(range(n_vars))
    ax_corr.set_yticklabels(cols, fontsize=8)
    cell_fs = max(5, min(8, 40 // max(n_vars, 1)))
    for i in range(n_vars):
        for j in range(n_vars):
            v = corr_matrix[i, j]
            tc = "white" if abs(v) > 0.5 else "black"
            ax_corr.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=cell_fs, color=tc
            )
    cbar1 = plt.colorbar(im1, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar1.set_label("Pearson r", fontsize=8)
    cbar1.ax.tick_params(labelsize=7)
    ax_corr.set_title("(a) Contemporaneous correlation", fontsize=9, pad=6)
    # Annotate mean absolute off-diagonal correlation
    mask = ~np.eye(n_vars, dtype=bool)
    mean_abs_corr = np.mean(np.abs(corr_matrix[mask]))
    ax_corr.text(
        0.5,
        -0.18,
        f"Mean |r| = {mean_abs_corr:.3f}",
        transform=ax_corr.transAxes,
        fontsize=7.5,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.3, alpha=0.9),
    )

    # Right: Lagged cross-correlation (max |corr| across lags)
    max_corr = np.zeros((n_vars, n_vars))
    best_lag_mat = np.zeros((n_vars, n_vars), dtype=int)

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                max_corr[i, j] = 1.0
                continue
            x = data[cols[i]].dropna().values
            y = data[cols[j]].dropna().values
            n = min(len(x), len(y))
            if n < tau_max + 10:
                continue
            best_r, best_l = 0.0, 0
            for lag in range(0, min(tau_max + 1, n // 3)):
                if lag == 0:
                    r = abs(np.corrcoef(x[:n], y[:n])[0, 1])
                else:
                    r = abs(np.corrcoef(x[: n - lag], y[lag:n])[0, 1])
                if r > best_r:
                    best_r = r
                    best_l = lag
            max_corr[i, j] = best_r
            best_lag_mat[i, j] = best_l

    im2 = ax_lag.imshow(max_corr, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax_lag.set_xticks(range(n_vars))
    ax_lag.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax_lag.set_yticks(range(n_vars))
    ax_lag.set_yticklabels(cols, fontsize=8)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            v = max_corr[i, j]
            lag = best_lag_mat[i, j]
            tc = "white" if v > 0.5 else "black"
            ax_lag.text(
                j,
                i,
                f"{v:.2f}\nlag {lag}",
                ha="center",
                va="center",
                fontsize=cell_fs,
                color=tc,
            )
    cbar2 = plt.colorbar(im2, ax=ax_lag, fraction=0.046, pad=0.04)
    cbar2.set_label("Max |cross-corr|", fontsize=8)
    cbar2.ax.tick_params(labelsize=7)
    ax_lag.set_title("(b) Lagged cross-correlation (best lag)", fontsize=9, pad=6)
    # Annotate strongest lagged pair
    best_i, best_j = 0, 1
    best_val = 0
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            if max_corr[i, j] > best_val:
                best_val = max_corr[i, j]
                best_i, best_j = i, j
    if best_val > 0:
        ax_lag.text(
            0.5,
            -0.18,
            f"Strongest: {cols[best_i]}$\\to${cols[best_j]} "
            f"(|r|={best_val:.2f}, lag={best_lag_mat[best_i, best_j]})",
            transform=ax_lag.transAxes,
            fontsize=7,
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.3, alpha=0.9
            ),
        )

    plt.tight_layout(w_pad=2.5)
    _save(fig, "correlation_and_lagged_crosscorrelation")


# ==============================================================================
# Figure 3: Spectral density
# ==============================================================================


def _plot_spectral_density(data, cols, _save):
    """Power spectral density per variable with visible peak annotations."""
    import matplotlib.pyplot as plt
    from scipy import signal as sp_signal

    n_plot = min(len(cols), 6)
    fig, axes = plt.subplots(n_plot, 1, figsize=(6.5, 1.6 * n_plot + 0.6), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for i, col in enumerate(cols[:n_plot]):
        ax = axes[i]
        s = data[col].dropna().values
        if len(s) < 50:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
            )
            ax.set_ylabel(col, fontsize=9)
            continue
        freqs, psd = sp_signal.periodogram(s - s.mean(), fs=1.0)
        ax.semilogy(freqs[1:], psd[1:], lw=0.8, color="#377EB8")
        ax.set_ylabel(col, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, 0.5)

        # Mark dominant frequency with high-contrast label
        if len(psd) > 1:
            peak_idx = np.argmax(psd[1:]) + 1
            peak_freq = freqs[peak_idx]
            peak_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
            peak_power_frac = (
                psd[peak_idx] / np.sum(psd[1:]) if np.sum(psd[1:]) > 0 else 0
            )
            ax.axvline(peak_freq, color="#E41A1C", lw=0.9, ls=":", alpha=0.9)
            # White background box for visibility over blue line
            ax.text(
                peak_freq + 0.02,
                0.82,
                f"Period={peak_period:.0f} ({peak_power_frac:.0%})",
                fontsize=7.5,
                color="#E41A1C",
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white",
                    ec="#E41A1C",
                    lw=0.4,
                    alpha=0.95,
                ),
            )

    axes[-1].set_xlabel("Frequency (cycles per time step)", fontsize=9)
    fig.suptitle(
        "Power spectral density per variable", fontweight="bold", y=1.01, fontsize=10
    )
    plt.tight_layout()
    _save(fig, "spectral_density")


# ==============================================================================
# Figure 4: Stationarity diagnostic
# ==============================================================================


def _plot_stationarity_diagnostic(data, cols, audit_evidence_dict, _save):
    """Rolling mean/std with ADF+KPSS verdict per variable."""
    import matplotlib.pyplot as plt

    n_plot = min(len(cols), 6)
    fig, axes = plt.subplots(n_plot, 1, figsize=(6.5, 1.8 * n_plot + 0.6), sharex=False)
    if n_plot == 1:
        axes = [axes]

    stat_diags = audit_evidence_dict.get("diagnostics", {}).get("stationarity", {})

    for i, col in enumerate(cols[:n_plot]):
        ax = axes[i]
        s = data[col].dropna()
        t = np.arange(len(s))
        vals_arr = s.values

        ax.plot(t, vals_arr, color="#377EB8", lw=0.5, alpha=0.7, rasterized=True)

        window = max(10, len(s) // 5)
        if window % 2 == 0:
            window += 1
        roll_mean = pd.Series(vals_arr).rolling(window, center=True).mean().values
        roll_std = pd.Series(vals_arr).rolling(window, center=True).std().values
        ax.plot(t, roll_mean, color="#E41A1C", lw=1.2, label="Rolling mean")
        ax.fill_between(
            t,
            roll_mean - roll_std,
            roll_mean + roll_std,
            color="#E41A1C",
            alpha=0.1,
            label="Rolling std",
        )

        d = stat_diags.get(col, {})
        adf_p = d.get("adf", {}).get("pvalue", None)
        kpss_p = d.get("kpss", {}).get("pvalue", None)
        # ADF: reject null (unit root) if p < 0.05 => stationary
        # KPSS: reject null (stationarity) if p < 0.05 => non-stationary
        adf_ok = d.get("adf", {}).get("is_stationary", None)
        kpss_ok = d.get("kpss", {}).get("is_stationary", None)
        # Derive from p-values if is_stationary not stored
        if adf_ok is None and adf_p is not None:
            adf_ok = adf_p < 0.05
        if kpss_ok is None and kpss_p is not None:
            kpss_ok = kpss_p > 0.05  # KPSS null = stationary

        if adf_ok is True and kpss_ok is True:
            verdict, vc = "Stationary: safe for ParCorr/Granger", "#4DAF4A"
        elif adf_ok is False and kpss_ok is False:
            verdict, vc = "Unit root: detrend before discovery", "#E41A1C"
        elif adf_ok is True and kpss_ok is False:
            verdict, vc = "Trend-stationary: consider detrending", "#FF7F00"
        elif adf_ok is False and kpss_ok is True:
            verdict, vc = "Difference-stationary: difference first", "#FF7F00"
        else:
            verdict, vc = "Inconclusive: inspect visually", "#FF7F00"

        ax.set_ylabel(col, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(
            0.99,
            0.92,
            verdict,
            transform=ax.transAxes,
            fontsize=7,
            color=vc,
            fontweight="bold",
            ha="right",
            va="top",
        )
        # Show break magnitude if available
        break_mag = d.get("break_magnitude_sigma", None)
        if break_mag is not None and break_mag > 0.5:
            ax.text(
                0.99,
                0.78,
                f"Break: {break_mag:.1f}$\\sigma$",
                transform=ax.transAxes,
                fontsize=6.5,
                color="#666666",
                ha="right",
                va="top",
            )

        if i == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        if i < n_plot - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time step", fontsize=9)

    fig.suptitle(
        "Stationarity diagnostic (rolling mean and std)",
        fontweight="bold",
        y=1.01,
        fontsize=10,
    )
    plt.tight_layout()
    _save(fig, "stationarity_diagnostic")


# ==============================================================================
# Figure 5: Sample size adequacy
# ==============================================================================


def _plot_sample_size_adequacy(data, cols, audit_evidence_dict, _save):
    """Effective sample size vs. method minimum requirements."""
    import matplotlib.pyplot as plt

    n_vars = len(cols)
    fig, ax = plt.subplots(figsize=(5.5, max(3.5, 0.5 * n_vars + 1.5)))

    persistence = audit_evidence_dict.get("diagnostics", {}).get("persistence", {})
    t_effs = []
    var_names = []
    for col in cols:
        p = persistence.get(col, {})
        if "t_eff" in p:
            t_effs.append(p["t_eff"])
            var_names.append(col)

    if not t_effs:
        t_effs = [float(len(data))] * n_vars
        var_names = cols[:]

    y = np.arange(len(var_names))
    ax.barh(
        y,
        t_effs,
        height=0.5,
        color="#377EB8",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )

    # Method requirement lines
    method_mins = {
        "ParCorr": 50,
        "CMIknn": 200,
        "GPDC": 500,
        "Granger": 75,
        "VARLiNGAM": 120,
    }
    colors_m = ["#4DAF4A", "#FF7F00", "#E41A1C", "#984EA3", "#A65628"]
    for idx, (method, tmin) in enumerate(method_mins.items()):
        ax.axvline(
            tmin,
            color=colors_m[idx],
            lw=0.9,
            ls="--",
            alpha=0.8,
            label=f"{method} (T_min={tmin})",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(var_names, fontsize=9)
    ax.set_xlabel("Effective sample size ($T_{eff}$)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(
        "Effective sample size vs. method requirements",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    # Legend below the plot (outside), no background, horizontal in 2 rows
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )

    # Value labels
    x_max = ax.get_xlim()[1]
    for i, v in enumerate(t_effs):
        if v > 0.7 * x_max:
            ax.text(
                v - x_max * 0.02,
                i,
                f"{v:.0f}",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                v + x_max * 0.01,
                i,
                f"{v:.0f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#333333",
            )

    plt.tight_layout()
    _save(fig, "sample_size_adequacy")


# ==============================================================================
# Figure 6: Assumption Deep Dive (per-assumption diagnostic + action guide)
# ==============================================================================


def _plot_assumption_deep_dive(
    data, cols, audit_evidence_dict, risk_profile_dict, _save
):
    """
    Per-assumption diagnostic figure. Each panel shows the evidence for one
    assumption violation and the recommended pre-processing action.

    Panels:
      (a) Nonstationarity: variance ratio over time (rolling var / global var)
      (b) Nonlinearity: residual diagnostic (linear fit residuals vs. fitted)
      (c) Persistence: integral timescale per variable (bar + decay curve)
      (d) Confounding: VIF per variable (bar chart with threshold)
      (e) Seasonality: spectral power concentration (bar per variable)
      (f) Irregularity: missing data pattern (heatmap of gaps)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_vars = len(cols)
    risks = risk_profile_dict.get("risks", {})
    diags = audit_evidence_dict.get("diagnostics", {})

    fig = plt.figure(figsize=(7.2, 8.5))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.32)

    # -- (a) Nonstationarity: rolling variance ratio --
    ax = fig.add_subplot(gs[0, 0])
    cmap_colors = plt.cm.Set2(np.linspace(0, 1, max(n_vars, 1)))
    for i, col in enumerate(cols[:4]):
        s = data[col].dropna().values
        if len(s) < 50:
            continue
        window = max(20, len(s) // 10)
        global_var = np.var(s)
        if global_var < 1e-12:
            continue
        roll_var = pd.Series(s).rolling(window, center=True).var().values
        ratio = roll_var / global_var
        ax.plot(
            np.arange(len(ratio)),
            ratio,
            color=cmap_colors[i],
            lw=0.7,
            alpha=0.85,
            label=col,
        )
    ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.axhline(2.0, color="#E41A1C", lw=0.6, ls=":", alpha=0.7)
    ax.set_xlabel("Time step", fontsize=8)
    ax.set_ylabel("Rolling var / global var", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    risk_val = risks.get("NonstationarityRisk", {}).get("mean", 0)
    action = "Detrend or difference" if risk_val >= 0.4 else "No action needed"
    ax.set_title(f"(a) Nonstationarity [risk={risk_val:.2f}]", fontsize=8, pad=3)
    ax.text(
        0.02,
        0.02,
        f"Action: {action}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        style="italic",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.2", fc="#FFFFEE", ec="#CCCCAA", lw=0.4, alpha=0.9
        ),
    )

    # -- (b) Nonlinearity: MI vs r^2 deviation from diagonal --
    ax = fig.add_subplot(gs[0, 1])
    pnl = diags.get("pairwise_nonlinearity", {})
    pairs = pnl.get("pairs", {})
    pair_names = list(pairs.keys())
    deviations = []
    for pname in pair_names:
        p = pairs[pname]
        r2 = p.get("r_squared", 0)
        mi = p.get("mi_estimate", 0)
        # Deviation: how much MI exceeds what r^2 would predict
        deviations.append(max(0, mi - r2))

    if deviations:
        y_pos = np.arange(len(pair_names))
        # Show top 6 pairs by deviation
        sorted_idx = np.argsort(deviations)[::-1][: min(6, len(deviations))]
        show_names = [pair_names[i][:10] for i in sorted_idx]
        show_devs = [deviations[i] for i in sorted_idx]
        show_nl = [
            str(pairs[pair_names[i]].get("is_nonlinear", False)).lower() == "true"
            for i in sorted_idx
        ]
        colors_b = ["#E41A1C" if nl else "#377EB8" for nl in show_nl]
        y_pos = np.arange(len(show_names))
        ax.barh(y_pos, show_devs, color=colors_b, height=0.5, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(show_names, fontsize=7)
        ax.set_xlabel("MI excess over $r^2$", fontsize=8)
        # Legend for bar colors
        from matplotlib.lines import Line2D as L2D

        ax.legend(
            handles=[
                L2D([0], [0], color="#E41A1C", lw=4, alpha=0.8, label="Nonlinear pair"),
                L2D([0], [0], color="#377EB8", lw=4, alpha=0.8, label="Linear pair"),
            ],
            frameon=True,
            fontsize=6.5,
            loc="lower right",
            fancybox=True,
            framealpha=0.85,
            edgecolor="#CCCCCC",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No pairwise data",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )
    ax.tick_params(labelsize=7)
    risk_val = risks.get("NonlinearityRisk", {}).get("mean", 0)
    action = "Use CMIknn CI test" if risk_val >= 0.35 else "ParCorr sufficient"
    ax.set_title(f"(b) Nonlinearity [risk={risk_val:.2f}]", fontsize=8, pad=3)
    ax.text(
        0.02,
        0.02,
        f"Action: {action}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        style="italic",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.2", fc="#FFFFEE", ec="#CCCCAA", lw=0.4, alpha=0.9
        ),
    )

    # -- (c) Persistence: T_eff per variable --
    ax = fig.add_subplot(gs[1, 0])
    persistence = diags.get("persistence", {})
    t_effs = []
    t_eff_names = []
    integral_taus = []
    for col in cols:
        p = persistence.get(col, {})
        if "t_eff" in p:
            t_effs.append(p["t_eff"])
            integral_taus.append(p.get("integral_tau", 0))
            t_eff_names.append(col)
    if not t_effs:
        t_effs = [float(len(data))] * n_vars
        integral_taus = [1.0] * n_vars
        t_eff_names = cols[:]

    y_pos = np.arange(len(t_eff_names))
    # Color by ratio: green if T_eff/T > 0.5, orange if > 0.2, red otherwise
    T = len(data)
    bar_colors = []
    for te in t_effs:
        ratio = te / T
        if ratio > 0.5:
            bar_colors.append("#4DAF4A")
        elif ratio > 0.2:
            bar_colors.append("#FF7F00")
        else:
            bar_colors.append("#E41A1C")
    ax.barh(y_pos, t_effs, color=bar_colors, height=0.5, alpha=0.8)
    ax.axvline(T, color="black", lw=0.8, ls="-", alpha=0.5, label=f"T={T}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(t_eff_names, fontsize=7.5)
    ax.set_xlabel("Effective sample size", fontsize=8)
    ax.tick_params(labelsize=7)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#4DAF4A", alpha=0.8, label="$T_{eff}/T > 0.5$"),
            Patch(facecolor="#FF7F00", alpha=0.8, label="$T_{eff}/T > 0.2$"),
            Patch(facecolor="#E41A1C", alpha=0.8, label="$T_{eff}/T \\leq 0.2$"),
        ],
        frameon=True,
        fontsize=6,
        loc="lower right",
        fancybox=True,
        framealpha=0.85,
        edgecolor="#CCCCCC",
    )
    # Annotate integral timescale
    for i, (te, it) in enumerate(zip(t_effs, integral_taus)):
        ax.text(
            te + T * 0.01,
            i,
            f"$\\tau_{{int}}$={it:.1f}",
            va="center",
            fontsize=6.5,
            color="#555555",
        )
    risk_val = risks.get("PersistenceRisk", {}).get("mean", 0)
    action = "Increase tau_max or subsample" if risk_val >= 0.5 else "Standard tau_max"
    ax.set_title(f"(c) Persistence [risk={risk_val:.2f}]", fontsize=8, pad=3)
    ax.text(
        0.02,
        0.02,
        f"Action: {action}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        style="italic",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.2", fc="#FFFFEE", ec="#CCCCAA", lw=0.4, alpha=0.9
        ),
    )

    # -- (d) Confounding: VIF per variable --
    ax = fig.add_subplot(gs[1, 1])
    confounding = diags.get("confounding", {})
    vif_data = confounding.get("vif", {})
    vif_names = []
    vif_vals = []
    for col in cols:
        if col in vif_data:
            vif_names.append(col)
            vif_vals.append(vif_data[col])
    if vif_vals:
        y_pos = np.arange(len(vif_names))
        bar_colors = [
            "#E41A1C" if v > 10 else "#FF7F00" if v > 5 else "#4DAF4A" for v in vif_vals
        ]
        ax.barh(y_pos, vif_vals, color=bar_colors, height=0.5, alpha=0.8)
        ax.axvline(5, color="#FF7F00", lw=0.7, ls="--", alpha=0.7, label="VIF=5")
        ax.axvline(10, color="#E41A1C", lw=0.7, ls="--", alpha=0.7, label="VIF=10")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(vif_names, fontsize=7.5)
        ax.set_xlabel("VIF", fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    else:
        ax.text(
            0.5,
            0.5,
            "VIF not computed",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )
    ax.tick_params(labelsize=7)
    risk_val = risks.get("ConfoundingRisk", {}).get("mean", 0)
    action = "Use LPCMCI (latent vars)" if risk_val >= 0.45 else "Standard methods OK"
    ax.set_title(f"(d) Confounding [risk={risk_val:.2f}]", fontsize=8, pad=3)
    ax.text(
        0.02,
        0.02,
        f"Action: {action}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        style="italic",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.2", fc="#FFFFEE", ec="#CCCCAA", lw=0.4, alpha=0.9
        ),
    )

    # -- (e) Seasonality: spectral power ratio per variable --
    ax = fig.add_subplot(gs[2, 0])
    seasonality = diags.get("seasonality", {})
    # Seasonality data is nested: seasonality["per_variable"][col]
    seas_per_var = seasonality.get("per_variable", seasonality)
    seas_names = []
    seas_ratios = []
    for col in cols:
        sd = seas_per_var.get(col, {})
        # Key is "spectral_peak_ratio" (not "spectral_ratio")
        ratio = sd.get("spectral_peak_ratio", sd.get("spectral_ratio", None))
        if ratio is not None:
            seas_names.append(col)
            seas_ratios.append(ratio)
    if seas_ratios:
        y_pos = np.arange(len(seas_names))
        bar_colors = [
            "#E41A1C" if r > 0.3 else "#FF7F00" if r > 0.15 else "#4DAF4A"
            for r in seas_ratios
        ]
        ax.barh(y_pos, seas_ratios, color=bar_colors, height=0.5, alpha=0.8)
        ax.axvline(0.15, color="#FF7F00", lw=0.7, ls="--", alpha=0.7)
        ax.axvline(0.30, color="#E41A1C", lw=0.7, ls="--", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(seas_names, fontsize=7.5)
        ax.set_xlabel("Spectral power ratio (peak / total)", fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No seasonality data",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )
    ax.tick_params(labelsize=7)
    risk_val = risks.get("SeasonalityRisk", {}).get("mean", 0)
    action = (
        "Deseasonalize before discovery"
        if risk_val >= 0.3
        else "No deseasonalization needed"
    )
    ax.set_title(f"(e) Seasonality [risk={risk_val:.2f}]", fontsize=8, pad=3)
    ax.text(
        0.02,
        0.02,
        f"Action: {action}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        style="italic",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.2", fc="#FFFFEE", ec="#CCCCAA", lw=0.4, alpha=0.9
        ),
    )

    # -- (f) Irregularity: missing data pattern --
    ax = fig.add_subplot(gs[2, 1])
    irregularity = diags.get("irregularity", {}).get("per_variable", {})
    if irregularity:
        miss_names = []
        miss_rates = []
        for col in cols:
            ir = irregularity.get(col, {})
            miss_names.append(col)
            miss_rates.append(ir.get("missing_rate", 0) * 100)

        y_pos = np.arange(len(miss_names))
        # Color by severity
        bar_colors = [
            "#E41A1C" if mr > 5 else "#FF7F00" if mr > 1 else "#4DAF4A"
            for mr in miss_rates
        ]
        ax.barh(y_pos, miss_rates, color=bar_colors, height=0.5, alpha=0.8)
        ax.axvline(
            5, color="#E41A1C", lw=0.7, ls="--", alpha=0.7, label=">5% (caution)"
        )
        ax.axvline(1, color="#FF7F00", lw=0.7, ls="--", alpha=0.7, label=">1% (minor)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(miss_names, fontsize=7.5)
        ax.set_xlabel("Missing data (%)", fontsize=8)
        ax.legend(
            frameon=True,
            fontsize=6.5,
            loc="lower right",
            fancybox=True,
            framealpha=0.85,
            edgecolor="#CCCCCC",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No irregularity data",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )
    ax.tick_params(labelsize=7)
    risk_val = risks.get("IrregularityRisk", {}).get("mean", 0)
    action = (
        "Impute or use robust methods" if risk_val >= 0.3 else "Missingness acceptable"
    )
    ax.set_title(f"(f) Irregularity [risk={risk_val:.2f}]", fontsize=8, pad=3)
    ax.text(
        0.02,
        0.02,
        f"Action: {action}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        style="italic",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.2", fc="#FFFFEE", ec="#CCCCAA", lw=0.4, alpha=0.9
        ),
    )

    fig.suptitle(
        "Assumption diagnostics: evidence and recommended actions",
        fontweight="bold",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    _save(fig, "assumption_deep_dive")


# ==============================================================================
# Figure 7: Preliminary causal graph (tigramite PCMCI+ with ParCorr)
# ==============================================================================


def _plot_dependency_network(data, cols, audit_evidence_dict, _save):
    """
    Run a fast PCMCI+ (ParCorr, tau_max=2) and plot the resulting causal
    graph using tigramite's native plot_graph. This gives researchers a
    preliminary causal structure estimate before running the full analysis
    with optimized parameters.

    Produces: dependency_network.pdf/.png
    """
    import matplotlib.pyplot as plt

    n_vars = len(cols)
    if n_vars < 2 or n_vars > 15:
        return  # Skip for trivial or very large systems

    try:
        from tigramite.pcmci import PCMCI
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite import plotting as tp
    except ImportError:
        return  # tigramite not available

    # Prepare data for tigramite
    data_array = data[cols].dropna().values
    if len(data_array) < 50:
        return

    var_names = cols[:]
    dataframe = pp.DataFrame(data_array, var_names=var_names)

    # Run fast PCMCI+ with ParCorr (linear, fast)
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    # Use tau_max=2 for speed (preliminary estimate)
    tau_max = min(2, len(data_array) // 20)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=0.05)

    graph = results["graph"]
    val_matrix = results["val_matrix"]

    # Plot using tigramite's native plot_graph (publication quality)
    fig, ax = tp.plot_graph(
        graph=graph,
        val_matrix=val_matrix,
        var_names=var_names,
        figsize=(6.0, 4.0),
        node_size=0.4,
        arrow_linewidth=4.0,
        curved_radius=0.2,
        node_label_size=10,
        link_label_fontsize=8,
        show_autodependency_lags=False,
    )
    plt.suptitle(
        "Preliminary causal graph (PCMCI+, ParCorr, $\\tau_{max}$="
        + str(tau_max)
        + ")",
        fontsize=9,
        fontweight="bold",
        y=0.98,
    )
    # Save manually since tp.plot_graph returns fig
    _save(fig, "dependency_network")
