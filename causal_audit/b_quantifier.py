"""
Module B: Uncertainty Quantifier (FIXED VERSION)

SCIENTIFIC FIX: Uses global diagnostic-to-risk parameters.

Aggregate diagnostics into calibrated composite risk indices with posteriors.
v0.2: Uses global parameters (works without family assignment).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import warnings

import numpy as np
from scipy.special import expit  # Logistic sigmoid

from .a_auditor import AuditEvidence


@dataclass
class RiskProfile:
    """Results from risk quantification (Module B output)."""

    # Schema metadata
    schema_version: str
    timestamp: str

    # Core risks (posterior means + 95% CrI)
    risks: Dict[str, Dict[str, float]]  # {risk_name: {mean, lower, upper}}

    # Risk ledger (top-3 contributing diagnostics per risk)
    risk_ledger: Dict[str, List[Dict[str, Any]]]

    # Joint covariance matrix (empirical)
    covariance_matrix: Optional[Dict[str, Any]]

    # Metadata
    n_risks: int
    risk_names: List[str]

    # Provenance
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RiskQuantifier:
    """
    Aggregate diagnostics into calibrated composite risk indices.

    v0.2: Uses global parameters - works without family assignment!
    """

    # Core risk names
    CORE_RISKS = [
        "NonstationarityRisk",
        "IrregularityRisk",
        "PersistenceRisk",
        "ConfoundingRisk",  # display: "Causal insufficiency"; proxy for latent confounders via VIF/Chow diagnostics
        "NonlinearityRisk",
        "SeasonalityRisk",
    ]

    def __init__(
        self, calibration_file: Optional[str] = None, random_seed: Optional[int] = None
    ):
        """
        Initialize quantifier.

        Args:
            calibration_file: Path to calib_v2.yaml (if available)
            random_seed: Random seed for reproducibility
        """
        self.calibration_file = calibration_file
        self.random_seed = random_seed

        # Load calibration parameters if available
        self.calibration_params = self._load_calibration()

        if random_seed is not None:
            np.random.seed(random_seed)

    def _load_calibration(self) -> Optional[Dict[str, Any]]:
        """Load calibration parameters from YAML file."""
        if self.calibration_file is None:
            return None

        try:
            import yaml

            with open(self.calibration_file, "r") as f:
                calib = yaml.safe_load(f)
            return calib
        except Exception as e:
            warnings.warn(f"Could not load calibration file: {e}")
            return None

    def quantify(
        self, audit_evidence: AuditEvidence, bootstrap_samples: int = 100
    ) -> RiskProfile:
        """
        Aggregate diagnostics into calibrated risk indices.

        Args:
            audit_evidence: Output from Module A
            bootstrap_samples: Number of bootstrap samples for uncertainty

        Returns:
            RiskProfile with posterior risk estimates
        """
        # Extract specific diagnostic values (not aggregated!)
        diagnostic_values = self._extract_diagnostic_values(audit_evidence)

        # Compute risk posteriors (mean + CrI)
        risks = {}
        risk_ledger = {}

        for risk_name in self.CORE_RISKS:
            risk_mean, risk_lower, risk_upper, ledger = self._compute_risk_posterior(
                risk_name,
                diagnostic_values,
                audit_evidence.t_eff,
                bootstrap_samples,
                n_samples=audit_evidence.n_samples,
            )

            risks[risk_name] = {
                "mean": float(risk_mean),
                "lower_95": float(risk_lower),
                "upper_95": float(risk_upper),
            }

            risk_ledger[risk_name] = ledger

        # Compute joint covariance (simplified for v0.2)
        covariance_matrix = {"note": "Empirical covariance computation deferred"}

        # Generate provenance
        from datetime import datetime, timezone

        provenance = audit_evidence.provenance.copy()
        provenance["module"] = "B_quantifier"
        provenance["version"] = "v0.2_fixed"
        provenance["uses_global_parameters"] = True

        return RiskProfile(
            schema_version="2.0.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            risks=risks,
            risk_ledger=risk_ledger,
            covariance_matrix=covariance_matrix,
            n_risks=len(self.CORE_RISKS),
            risk_names=self.CORE_RISKS,
            provenance=provenance,
        )

    def _extract_diagnostic_values(
        self, audit_evidence: AuditEvidence
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract specific diagnostic values (not aggregated effect sizes).

        Returns dict mapping risk_name → {diagnostic_name: value}
        """
        diagnostics_dict = audit_evidence.diagnostics
        extracted = {}

        # NonstationarityRisk diagnostics
        nonstat_diags = {
            "break_magnitude": [],
            "drift_slope": [],
            "adf_log_p": [],
            "kpss_log_p": [],
        }

        for var, stats in diagnostics_dict.get("stationarity", {}).items():
            if "error" not in stats:
                nonstat_diags["break_magnitude"].append(
                    stats.get("break_magnitude_sigma", 0.0)
                )
                nonstat_diags["drift_slope"].append(
                    abs(stats.get("drift_slope_normalized", 0.0))
                )
                nonstat_diags["adf_log_p"].append(
                    stats.get("adf", {}).get("log_p_zscore", 0.0)
                )
                nonstat_diags["kpss_log_p"].append(
                    stats.get("kpss", {}).get("log_p_zscore", 0.0)
                )

        # Average across variables
        extracted["NonstationarityRisk"] = {
            k: np.mean(v) if len(v) > 0 else 0.0 for k, v in nonstat_diags.items()
        }

        # IrregularityRisk diagnostics
        irregular_diags = {
            "gap_cv": diagnostics_dict.get("irregularity", {})
            .get("global", {})
            .get("gap_cv", 0.0),
            "missing_rate": 0.0,
            "mnar_wald_stat": 0.0,
        }

        missing_rates = []
        mnar_stats = []
        for var, stats in (
            diagnostics_dict.get("irregularity", {}).get("per_variable", {}).items()
        ):
            missing_rates.append(stats.get("missing_rate", 0.0))
            if "mnar_seasonal" in stats:
                mnar_stats.append(stats["mnar_seasonal"].get("log_p_zscore", 0.0))

        irregular_diags["missing_rate"] = (
            np.mean(missing_rates) if len(missing_rates) > 0 else 0.0
        )
        irregular_diags["mnar_wald_stat"] = (
            np.mean(mnar_stats) if len(mnar_stats) > 0 else 0.0
        )

        extracted["IrregularityRisk"] = irregular_diags

        # PersistenceRisk diagnostics
        t_eff_ratios = []
        integral_taus = []
        for var, stats in diagnostics_dict.get("persistence", {}).items():
            if "error" not in stats:
                t_eff_ratios.append(stats.get("t_eff_ratio", 1.0))
                integral_taus.append(stats.get("integral_tau", 0.0))

        extracted["PersistenceRisk"] = {
            "t_eff_ratio": np.mean(t_eff_ratios) if len(t_eff_ratios) > 0 else 1.0,
            "integral_tau": np.mean(integral_taus) if len(integral_taus) > 0 else 0.0,
        }

        # ConfoundingRisk diagnostics (displayed as "Causal insufficiency")
        # These are proxy indicators for potential latent confounders; true confounding
        # is unidentifiable from observational data alone (Spirtes et al., 2000).
        chow_stats = []
        vif_vals = []

        for var, stats in (
            diagnostics_dict.get("confounding", {}).get("chow_tests", {}).items()
        ):
            if isinstance(stats, dict):
                # Handle both 'f_statistic' and 'chow_f_statistic' key names
                if "chow_f_statistic" in stats:
                    chow_stats.append(stats["chow_f_statistic"])
                elif "f_statistic" in stats:
                    chow_stats.append(stats["f_statistic"])

        vif_stats = diagnostics_dict.get("confounding", {}).get("vif", {})
        if isinstance(vif_stats, dict) and "error" not in vif_stats:
            for var, vif_val in vif_stats.items():
                if var.startswith("_"):  # Skip metadata keys
                    continue
                if isinstance(vif_val, (int, float)):
                    # Handle inf values - cap at 1e6 for numerical stability
                    if np.isinf(vif_val):
                        vif_vals.append(1e6)  # Extreme but finite
                    elif not np.isnan(vif_val):
                        vif_vals.append(min(vif_val, 1e6))  # Cap at 1e6

        # Compute vif_max, handling empty case
        if len(vif_vals) > 0:
            vif_max = np.max(vif_vals)
        else:
            # Check if we have metadata indicating near-singularity
            if "_condition_number" in vif_stats:
                vif_max = 1e6  # Extreme multicollinearity detected
            else:
                vif_max = 1.0  # Default (no VIF data)

        extracted["ConfoundingRisk"] = {
            "chow_f_stat": np.mean(chow_stats) if len(chow_stats) > 0 else 1.0,
            "residual_variance_ratio": 1.0,  # Not yet implemented in auditor
            "vif_max": vif_max,
        }

        # NonlinearityRisk diagnostics — from pairwise_nonlinearity and basic nonlinearity
        # Uses the RESET-based pairwise test (fraction of nonlinear pairs) and
        # the univariate RF vs. linear ΔRMSE (average across variables).
        # Seasonality is now a separate SeasonalityRisk dimension.
        nonlin_diags = {
            "fraction_nonlinear_pairs": 0.0,
            "mean_delta_rmse": 0.0,
        }

        pairwise_nl = diagnostics_dict.get("pairwise_nonlinearity", {})
        if isinstance(pairwise_nl, dict) and "fraction_nonlinear" in pairwise_nl:
            nonlin_diags["fraction_nonlinear_pairs"] = float(
                pairwise_nl.get("fraction_nonlinear", 0.0)
            )

        basic_nl = diagnostics_dict.get("nonlinearity", {})
        delta_rmse_vals = []
        for var, stats in basic_nl.items():
            if isinstance(stats, dict) and "delta_rmse_relative" in stats:
                delta_rmse_vals.append(max(0.0, stats["delta_rmse_relative"]))
        if delta_rmse_vals:
            nonlin_diags["mean_delta_rmse"] = float(np.mean(delta_rmse_vals))

        extracted["NonlinearityRisk"] = nonlin_diags

        # SeasonalityRisk diagnostics — dedicated risk for trend+seasonal data.
        # High SeasonalityRisk means the data has strong periodic components that
        # violate the stationarity assumption of ParCorr/Granger. The recommended
        # action is detrending/deseasonalization before discovery, not abstention.
        seasonality = diagnostics_dict.get("seasonality", {})
        seasonal_diags = {
            "mean_spectral_ratio": 0.0,
            "fraction_with_seasonality": 0.0,
            "fraction_with_trend": 0.0,
        }
        if isinstance(seasonality, dict):
            global_s = seasonality.get("global", {})
            seasonal_diags["mean_spectral_ratio"] = float(
                global_s.get("mean_spectral_ratio", 0.0)
            )
            seasonal_diags["fraction_with_seasonality"] = float(
                global_s.get("fraction_with_seasonality", 0.0)
            )
            seasonal_diags["fraction_with_trend"] = float(
                global_s.get("fraction_with_trend", 0.0)
            )

        extracted["SeasonalityRisk"] = seasonal_diags

        return extracted

    def _compute_risk_posterior(
        self,
        risk_name: str,
        diagnostic_values: Dict[str, Dict[str, float]],
        t_eff: Dict[str, float],
        bootstrap_samples: int,
        n_samples: int = 365,
    ) -> Tuple[float, float, float, List[Dict[str, Any]]]:
        """
        Compute risk posterior using global parameters.

        Formula: logit(risk) = alpha + sum(weight_i * diagnostic_i)

        Args:
            n_samples: actual series length T (used for uncertainty inflation).

        Returns (mean, lower_95, upper_95, ledger).
        """
        # Get diagnostics for this risk
        diags = diagnostic_values.get(risk_name, {})

        # Load global parameters (if available)
        if self.calibration_params and "global_parameters" in self.calibration_params:
            global_params = self.calibration_params["global_parameters"]["risks"].get(
                risk_name, {}
            )
            alpha = global_params.get("alpha", -1.5)
            diagnostic_weights_dict = global_params.get("diagnostic_weights", {})

            # Extract weights
            weights = {}
            for diag_name, diag_info in diagnostic_weights_dict.items():
                if isinstance(diag_info, dict):
                    weights[diag_name] = diag_info.get("weight", 1.0)
                else:
                    weights[diag_name] = float(diag_info)
        else:
            # Fallback heuristics
            alpha = -1.5
            weights = {k: 1.0 for k in diags.keys()}

        # Compute logit score
        logit_score = alpha
        contributions = {}

        for diag_name, diag_value in diags.items():
            if diag_value is None:
                continue  # Skip None values
            weight = weights.get(diag_name, 0.0)
            contribution = weight * diag_value
            logit_score += contribution
            contributions[diag_name] = contribution

        # Apply sigmoid
        risk_mean = expit(logit_score)

        # Bootstrap for UI (resample diagnostic values)
        bootstrap_risks = []
        for _ in range(bootstrap_samples):
            # Add noise to diagnostics (simplified bootstrap)
            boot_logit = alpha
            for diag_name, diag_value in diags.items():
                if diag_value is None:
                    continue  # Skip None values
                weight = weights.get(diag_name, 0.0)
                # Add Gaussian noise (std = 0.1 * value)
                noise = np.random.normal(0, max(0.1 * abs(diag_value), 0.05))
                boot_logit += weight * (diag_value + noise)

            boot_risk = expit(boot_logit)
            bootstrap_risks.append(boot_risk)

        # Compute UI with T_eff-aware widening
        mean_t_eff = np.mean(list(t_eff.values())) if len(t_eff) > 0 else 100.0

        # Inflate uncertainty by sqrt(T / T_eff) using actual series length
        T_actual = float(n_samples)
        uncertainty_inflation = np.sqrt(T_actual / mean_t_eff)

        risk_std = np.std(bootstrap_risks) * uncertainty_inflation
        risk_lower = max(0.0, risk_mean - 1.96 * risk_std)
        risk_upper = min(1.0, risk_mean + 1.96 * risk_std)

        # Generate risk ledger (top-3 contributors)
        sorted_contribs = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        ledger = []
        for diag_name, contrib in sorted_contribs:
            ledger.append(
                {
                    "diagnostic": diag_name,
                    "value": float(diags[diag_name]),
                    "weight": float(weights.get(diag_name, 0.0)),
                    "contribution": float(contrib),
                }
            )

        return risk_mean, risk_lower, risk_upper, ledger
