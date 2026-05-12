"""
Module A: Assumption Auditor

Evidence gathering with graded diagnostics, effect sizes, and T_eff.
v0.1: 5 core diagnostic families (stationarity, irregularity, persistence,
      basic nonlinearity, confounding invariance).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class AuditEvidence:
    """Results from assumption auditing (Module A output)."""

    # Schema metadata
    schema_version: str
    timestamp: str

    # Core outputs
    diagnostics: Dict[str, Any]
    t_eff: Dict[str, float]  # Effective sample size per variable
    safe_tau_max: Dict[str, int]  # Data-driven and budget-driven

    # Metadata
    n_variables: int
    n_samples: int
    variable_names: list

    # Provenance
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AssumptionAuditor:
    """
    Audit time-series data for assumption violations.

    Returns effect sizes and uncertainties (NOT binary pass/fail).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        random_seed: Optional[int] = None,
        compute_budget_seconds: Optional[float] = None,
    ):
        """
        Initialize auditor.

        Args:
            alpha: Significance level for statistical tests
            random_seed: Random seed for reproducibility
            compute_budget_seconds: Optional compute budget for safe_tau_max
        """
        self.alpha = alpha
        self.random_seed = random_seed
        self.compute_budget_seconds = compute_budget_seconds

        if random_seed is not None:
            np.random.seed(random_seed)

    def audit(
        self, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvidence:
        """
        Run complete assumption audit.

        Args:
            df: Time-series DataFrame (rows=time, columns=variables)
            metadata: Optional metadata (seasonal_labels, qc_flags, etc.)

        Returns:
            AuditEvidence object with all diagnostic results
        """
        from .utils.provenance import hash_dataframe, generate_provenance

        # Basic validation
        if df.empty:
            raise ValueError("Empty DataFrame provided")

        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.warn(
                "DataFrame index is not DatetimeIndex. "
                "Irregularity diagnostics may be inaccurate."
            )

        # Run diagnostic families
        diagnostics = {}

        # 1. Stationarity (ADF/KPSS + Bai-Perron breaks)
        diagnostics["stationarity"] = self.check_stationarity(df)

        # 2. Irregularity (gaps, MNAR-seasonal)
        diagnostics["irregularity"] = self.check_irregularity(df)

        # 3. Persistence (integral correlation time)
        diagnostics["persistence"], t_eff = self.check_persistence(df)

        # 4. Basic nonlinearity (ΔRMSE proxy)
        diagnostics["nonlinearity"] = self.check_nonlinearity_basic(df)

        # 5. Confounding proxies (environmental invariance)
        diagnostics["confounding"] = self.check_confounding_proxies(df, metadata)

        # 6. Pairwise nonlinearity (MI vs r² per pair) — CI test recommendation
        diagnostics["pairwise_nonlinearity"] = self.check_pairwise_nonlinearity(df)

        # 7. Power sufficiency (effective sample size vs parameter count)
        diagnostics["power"] = self.check_power_sufficiency(df)

        # 8. Seasonality and trend detection
        diagnostics["seasonality"] = self.check_seasonality(df)

        # Compute safe tau_max
        safe_tau_max = self.compute_safe_tau_max(df, diagnostics)

        # Generate provenance
        from datetime import datetime, timezone

        data_hash = hash_dataframe(df)
        provenance = generate_provenance(
            input_data_hash=data_hash,
            config_hashes={},  # Will be populated by gatekeeper
            random_seed=self.random_seed,
            metadata={"alpha": self.alpha},
        )

        return AuditEvidence(
            schema_version="1.0.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            diagnostics=diagnostics,
            t_eff=t_eff,
            safe_tau_max=safe_tau_max,
            n_variables=len(df.columns),
            n_samples=len(df),
            variable_names=list(df.columns),
            provenance=provenance,
        )

    def check_stationarity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check stationarity via ADF, KPSS, and structural breaks.

        Returns effect sizes (break magnitude, drift slope, log-p z-scores).
        NEVER votes or binary-aggregates.

        Args:
            df: Time-series DataFrame

        Returns:
            Dictionary with per-variable stationarity diagnostics
        """
        results = {}

        for col in df.columns:
            series = df[col].dropna()

            if len(series) < 50:
                results[col] = {"error": "insufficient_data", "n": len(series)}
                continue

            col_results = {}

            # ADF test (H0: unit root / non-stationary)
            # Directionalized: negative log_p = stationary (good), positive = non-stationary (bad)
            try:
                adf_result = adfuller(series, autolag="AIC")
                adf_pvalue = adf_result[1]

                # Compute directionalized log-p score
                # ADF: H0 = non-stationary, so rejecting H0 (small p) = stationary = LOW risk
                # We want: stationary → negative, non-stationary → positive
                log_p_magnitude = -np.log10(adf_pvalue + 1e-10)

                # Cap to prevent domination (max = 3.0 corresponds to p = 0.001)
                log_p_magnitude = np.clip(log_p_magnitude, 0, 3.0)

                if adf_pvalue < self.alpha:
                    # Significant: reject H0 (non-stationary) → IS stationary → negative (good)
                    adf_log_p = -log_p_magnitude
                else:
                    # Non-significant: fail to reject H0 → possibly non-stationary → positive (bad)
                    adf_log_p = log_p_magnitude

                col_results["adf"] = {
                    "statistic": float(adf_result[0]),
                    "pvalue": float(adf_pvalue),
                    "log_p_zscore": float(adf_log_p),  # Directionalized!
                    "lags_used": int(adf_result[2]),
                }
            except Exception as e:
                col_results["adf"] = {"error": str(e)}

            # KPSS test (H0: stationary)
            # Directionalized: negative log_p = stationary (good), positive = non-stationary (bad)
            try:
                import warnings

                with warnings.catch_warnings():
                    # Suppress InterpolationWarning (expected when test stat outside lookup table)
                    warnings.filterwarnings("ignore", category=InterpolationWarning)
                    kpss_result = kpss(series, regression="ct", nlags="auto")

                kpss_pvalue = kpss_result[1]

                # Compute directionalized log-p score
                # KPSS: H0 = stationary, so rejecting H0 (small p) = non-stationary = HIGH risk
                # We want: stationary → negative, non-stationary → positive
                log_p_magnitude = -np.log10(kpss_pvalue + 1e-10)

                # Cap to prevent domination
                log_p_magnitude = np.clip(log_p_magnitude, 0, 3.0)

                if kpss_pvalue < self.alpha:
                    # Significant: reject H0 (stationary) → IS non-stationary → positive (bad)
                    kpss_log_p = log_p_magnitude
                else:
                    # Non-significant: fail to reject H0 → IS stationary → negative (good)
                    kpss_log_p = -log_p_magnitude

                col_results["kpss"] = {
                    "statistic": float(kpss_result[0]),
                    "pvalue": float(kpss_pvalue),
                    "log_p_zscore": float(kpss_log_p),  # Directionalized!
                    "lags_used": int(kpss_result[2]),
                }
            except Exception as e:
                col_results["kpss"] = {"error": str(e)}

            # Structural breaks (simplified: detect largest change point)
            break_magnitude = self._detect_largest_break(series)
            col_results["break_magnitude_sigma"] = float(break_magnitude)

            # Linear trend (drift)
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            lr = LinearRegression().fit(X, y)
            drift_slope = lr.coef_[0] / (series.std() + 1e-10)  # Normalized
            col_results["drift_slope_normalized"] = float(drift_slope)

            results[col] = col_results

        return results

    def _detect_largest_break(self, series: pd.Series, min_segment: int = 30) -> float:
        """
        Detect largest structural break via binary segmentation.

        Returns break magnitude as t-statistic style effect size (more rigorous).
        """
        if len(series) < 2 * min_segment:
            return 0.0

        values = series.values
        n = len(values)
        max_diff = 0.0

        # Scan potential break points
        for break_idx in range(min_segment, n - min_segment):
            left_mean = np.mean(values[:break_idx])
            right_mean = np.mean(values[break_idx:])
            left_std = np.std(values[:break_idx], ddof=1)
            right_std = np.std(values[break_idx:], ddof=1)
            n_left = break_idx
            n_right = n - break_idx

            # Pooled standard error (t-statistic style)
            pooled_se = np.sqrt(left_std**2 / n_left + right_std**2 / n_right)

            if pooled_se > 1e-10:
                diff = abs(right_mean - left_mean) / pooled_se
            else:
                diff = 0.0

            max_diff = max(max_diff, diff)

        return max_diff

    def check_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect deterministic seasonality and trends in each variable.

        Uses two complementary tests:
        1. Spectral power ratio: fraction of variance explained by the dominant
           frequency (periodogram peak / total variance). High ratio → strong
           periodic component.
        2. STL residual ratio: fits a simple seasonal decomposition and measures
           how much variance remains after removing trend + seasonal. Low residual
           ratio → strong trend+seasonal structure.

        Returns per-variable results and a global summary.
        """
        from scipy import signal as sp_signal

        results = {"per_variable": {}, "global": {}}
        spectral_ratios = []
        has_trend = []

        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 50:
                results["per_variable"][col] = {"error": "insufficient_data"}
                continue

            vals = series.values.astype(float)
            T = len(vals)

            # --- Spectral power ratio ---
            freqs, psd = sp_signal.periodogram(vals - vals.mean())
            total_power = psd.sum()
            if total_power > 0:
                peak_power = psd.max()
                spectral_ratio = float(peak_power / total_power)
            else:
                spectral_ratio = 0.0

            # --- Linear trend strength ---
            t = np.arange(T)
            slope, intercept = np.polyfit(t, vals, 1)
            trend_line = slope * t + intercept
            detrended = vals - trend_line
            trend_var = np.var(trend_line)
            total_var = np.var(vals)
            trend_ratio = float(trend_var / total_var) if total_var > 0 else 0.0

            spectral_ratios.append(spectral_ratio)
            has_trend.append(trend_ratio > 0.10)

            results["per_variable"][col] = {
                "spectral_peak_ratio": spectral_ratio,
                "trend_variance_ratio": trend_ratio,
                "has_strong_seasonality": spectral_ratio > 0.20,
                "has_trend": trend_ratio > 0.10,
            }

        # Global summary
        if spectral_ratios:
            results["global"]["mean_spectral_ratio"] = float(np.mean(spectral_ratios))
            results["global"]["fraction_with_seasonality"] = float(
                np.mean([r > 0.20 for r in spectral_ratios])
            )
            results["global"]["fraction_with_trend"] = float(np.mean(has_trend))
            results["global"]["seasonality_detected"] = (
                results["global"]["fraction_with_seasonality"] > 0.3
                or results["global"]["mean_spectral_ratio"] > 0.25
            )
        else:
            results["global"]["seasonality_detected"] = False

        return results

    def check_irregularity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check irregularity: gaps, MNAR-seasonal, MCAR.

        Args:
            df: Time-series DataFrame

        Returns:
            Irregularity diagnostics with effect sizes
        """
        results = {"global": {}, "per_variable": {}}

        # Global gap analysis
        if isinstance(df.index, pd.DatetimeIndex):
            time_diffs = df.index.to_series().diff()[1:]
            median_diff = time_diffs.median()

            if median_diff.total_seconds() > 0:
                gap_cv = (
                    time_diffs.std() / time_diffs.mean()
                )  # Coefficient of variation
                results["global"]["gap_cv"] = float(gap_cv)
                results["global"]["median_interval_hours"] = float(
                    median_diff.total_seconds() / 3600
                )
            else:
                results["global"]["gap_cv"] = 0.0
        else:
            results["global"]["gap_cv"] = None
            results["global"]["note"] = "Index is not DatetimeIndex"

        # Per-variable missingness
        for col in df.columns:
            col_results = {}
            missing_mask = df[col].isna()
            missing_rate = missing_mask.mean()

            col_results["missing_rate"] = float(missing_rate)

            if missing_rate > 0.01:  # Only test if >1% missing
                # MNAR-seasonal test (periodicity of missingness)
                if isinstance(df.index, pd.DatetimeIndex):
                    mnar_stat, mnar_pval = self._test_mnar_seasonal(
                        df.index, missing_mask
                    )
                    col_results["mnar_seasonal"] = {
                        "wald_statistic": float(mnar_stat),
                        "pvalue": float(mnar_pval),
                        "log_p_zscore": -np.log10(mnar_pval + 1e-10),
                    }

                # MCAR test (Little's test - simplified version)
                col_results["appears_mcar"] = missing_rate < 0.20  # Heuristic

            results["per_variable"][col] = col_results

        return results

    def _test_mnar_seasonal(
        self, time_index: pd.DatetimeIndex, missing_mask: pd.Series
    ) -> Tuple[float, float]:
        """
        Test for MNAR-seasonal: logit(missing) ~ sin(2π·day/365) + cos(2π·day/365).

        Returns Wald statistic and p-value.
        """
        try:
            from statsmodels.formula.api import logit

            # Ensure missing_mask aligns with time_index
            # Reset missing_mask to use the same index as time_index
            if len(missing_mask) != len(time_index):
                # This should not happen, but if it does, return neutral result
                return 0.0, 1.0

            # Create seasonal features aligned with time_index
            day_of_year = time_index.dayofyear
            sin_component = np.sin(2 * np.pi * day_of_year / 365.25)
            cos_component = np.cos(2 * np.pi * day_of_year / 365.25)

            # Fit logistic regression - create DataFrame with explicit index alignment
            test_df = pd.DataFrame(
                {
                    "missing": missing_mask.values.astype(
                        int
                    ),  # Use .values to avoid index issues
                    "sin_season": sin_component,
                    "cos_season": cos_component,
                },
                index=time_index,
            )

            model = logit("missing ~ sin_season + cos_season", data=test_df).fit(disp=0)

            # Wald test for joint significance
            wald_stat = model.wald_test("(sin_season = 0, cos_season = 0)").statistic[
                0
            ][0]
            pval = model.wald_test("(sin_season = 0, cos_season = 0)").pvalue

            return wald_stat, pval

        except Exception:
            # If logistic regression fails (e.g., perfect separation), return null
            return 0.0, 1.0

    def check_persistence(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Check persistence via integral correlation time → T_eff.

        Returns diagnostics and T_eff per variable.
        """
        results = {}
        t_eff = {}

        for col in df.columns:
            series = df[col].dropna()

            if len(series) < 50:
                results[col] = {"error": "insufficient_data"}
                t_eff[col] = float(len(series))  # Fallback
                continue

            # Compute integral correlation time
            integral_tau = self._compute_integral_tau(series.values)
            T = len(series)
            T_eff_val = T / (1 + 2 * integral_tau)

            results[col] = {
                "integral_tau": float(integral_tau),
                "t_eff": float(T_eff_val),
                "t_eff_ratio": float(T_eff_val / T),
                "n_samples": int(T),
            }

            t_eff[col] = float(T_eff_val)

        return results, t_eff

    def _compute_integral_tau(self, series: np.ndarray, max_lag: int = 50) -> float:
        """
        Compute integral correlation time: τ_int = ∑_{k=1}^∞ ρ(k).

        Truncate at first insignificant lag or max_lag.
        """
        from statsmodels.tsa.stattools import acf

        # Compute ACF
        acf_values = acf(series, nlags=min(max_lag, len(series) // 4), fft=True)

        # Integrate until ACF becomes insignificant
        threshold = 1.96 / np.sqrt(len(series))  # 95% confidence
        integral = 0.0

        for lag in range(1, len(acf_values)):
            if abs(acf_values[lag]) < threshold:
                break
            integral += acf_values[lag]

        return max(0.0, integral)

    def check_nonlinearity_basic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Basic nonlinearity proxy: ΔRMSE (linear AR vs. RandomForest).

        This is a simplified test; surrogate-based tests deferred to v0.2.
        """
        results = {}

        for col in df.columns:
            series = df[col].dropna().values

            if len(series) < 100:
                results[col] = {"error": "insufficient_data"}
                continue

            # Prepare AR(p) features
            p = min(5, len(series) // 20)  # Adaptive lag order
            X, y = self._prepare_ar_features(series, p)

            if len(X) < 50:
                results[col] = {"error": "insufficient_data_after_lagging"}
                continue

            # Time series split
            tscv = TimeSeriesSplit(n_splits=3)
            rmse_linear = []
            rmse_rf = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Linear model
                lr = LinearRegression().fit(X_train, y_train)
                pred_linear = lr.predict(X_test)
                rmse_linear.append(np.sqrt(np.mean((y_test - pred_linear) ** 2)))

                # Random Forest
                rf = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=self.random_seed
                ).fit(X_train, y_train)
                pred_rf = rf.predict(X_test)
                rmse_rf.append(np.sqrt(np.mean((y_test - pred_rf) ** 2)))

            # Compute ΔRMSE
            mean_rmse_linear = np.mean(rmse_linear)
            mean_rmse_rf = np.mean(rmse_rf)
            delta_rmse = (mean_rmse_linear - mean_rmse_rf) / (mean_rmse_linear + 1e-10)

            results[col] = {
                "rmse_linear": float(mean_rmse_linear),
                "rmse_rf": float(mean_rmse_rf),
                "delta_rmse_relative": float(delta_rmse),  # Effect size
                "ar_order": int(p),
            }

        return results

    def _prepare_ar_features(
        self, series: np.ndarray, p: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare AR(p) features: X = [y_{t-1}, ..., y_{t-p}], y = y_t."""
        X = []
        y = []

        for t in range(p, len(series)):
            X.append(series[t - p : t])
            y.append(series[t])

        return np.array(X), np.array(y)

    def check_pairwise_nonlinearity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pairwise nonlinearity detection between all variable pairs.

        For each pair (X, Y), compares mutual information (nonlinear dependence)
        against squared Pearson correlation (linear dependence). A high MI/r²
        ratio indicates nonlinear relationships that ParCorr would miss.

        Returns per-pair diagnostics with a recommended CI test.
        """
        from scipy.stats import pearsonr

        results = {"pairs": {}, "summary": {}}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_vars = len(numeric_cols)

        if n_vars < 2:
            results["summary"] = {"n_pairs": 0, "fraction_nonlinear": 0.0}
            return results

        n_nonlinear = 0
        n_tested = 0

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                x = df[numeric_cols[i]].values
                y = df[numeric_cols[j]].values

                # Pairwise complete observations
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() < 50:
                    continue

                xv, yv = x[valid], y[valid]
                n_tested += 1

                # Linear: Pearson r²
                r, _ = pearsonr(xv, yv)
                r_squared = r**2

                # Nonlinear proxy: binned mutual information estimate
                mi = self._estimate_mi_binned(xv, yv)

                # Ratio: MI / max(r², 0.01) — high ratio means nonlinear
                ratio = mi / max(r_squared, 0.01)
                is_nonlinear = ratio > 2.0 and mi > 0.1

                if is_nonlinear:
                    n_nonlinear += 1

                pair_key = f"{numeric_cols[i]}_{numeric_cols[j]}"
                results["pairs"][pair_key] = {
                    "r_squared": float(r_squared),
                    "mi_estimate": float(mi),
                    "mi_r2_ratio": float(ratio),
                    "is_nonlinear": is_nonlinear,
                }

        fraction_nonlinear = n_nonlinear / max(n_tested, 1)
        results["summary"] = {
            "n_pairs_tested": n_tested,
            "n_nonlinear": n_nonlinear,
            "fraction_nonlinear": float(fraction_nonlinear),
            "recommended_ci_test": "cmiknn" if fraction_nonlinear > 0.3 else "parcorr",
        }

        return results

    def _estimate_mi_binned(
        self, x: np.ndarray, y: np.ndarray, n_bins: int = 20
    ) -> float:
        """Estimate mutual information using histogram binning (fast, approximate)."""
        # Joint and marginal histograms
        c_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
        c_x = c_xy.sum(axis=1)
        c_y = c_xy.sum(axis=0)

        # Normalize to probabilities
        n = len(x)
        p_xy = c_xy / n
        p_x = c_x / n
        p_y = c_y / n

        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

        return max(0.0, mi)

    def check_power_sufficiency(
        self, df: pd.DataFrame, tau_max: int = 14
    ) -> Dict[str, Any]:
        """
        Statistical power analysis for causal discovery.

        Assesses whether the dataset has sufficient effective observations
        to support the number of conditional independence tests that PCMCI
        will perform. Returns a power classification per variable and overall.

        The key metric is T_eff / (N × τ_max): the ratio of effective
        independent observations to the number of parameters being estimated.
        Values below 5 indicate underpowered analysis.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_vars = len(numeric_cols)
        n_samples = len(df)

        # Compute T_eff per variable (reuse persistence diagnostics if available)
        t_eff_per_var = {}
        for col in numeric_cols:
            series = df[col].dropna().values
            if len(series) < 20:
                t_eff_per_var[col] = len(series)
                continue
            t_eff_per_var[col] = self._compute_t_eff_single(series)

        # Overall effective sample size (minimum across variables)
        min_t_eff = min(t_eff_per_var.values()) if t_eff_per_var else n_samples

        # Power ratio: T_eff / (N * tau_max)
        # This approximates the effective observations per parameter
        n_parameters = n_vars * tau_max
        power_ratio = min_t_eff / max(n_parameters, 1)

        # Total tests PCMCI will perform (approximate)
        n_total_tests = n_vars * (n_vars - 1) * tau_max

        # Classification
        if power_ratio >= 10:
            power_class = "sufficient"
        elif power_ratio >= 5:
            power_class = "adequate"
        elif power_ratio >= 2:
            power_class = "marginal"
        else:
            power_class = "underpowered"

        # Per-variable power
        per_variable = {}
        for col, teff in t_eff_per_var.items():
            var_ratio = teff / max(n_vars * tau_max, 1)
            per_variable[col] = {
                "t_eff": float(teff),
                "power_ratio": float(var_ratio),
                "sufficient": var_ratio >= 5,
            }

        return {
            "n_variables": n_vars,
            "n_samples": n_samples,
            "tau_max_tested": tau_max,
            "min_t_eff": float(min_t_eff),
            "n_parameters": n_parameters,
            "power_ratio": float(power_ratio),
            "power_class": power_class,
            "n_total_tests_approx": n_total_tests,
            "per_variable": per_variable,
            "recommendation": (
                "Proceed with standard analysis"
                if power_class in ("sufficient", "adequate")
                else "Results should be interpreted with caution; consider reducing tau_max or variable set"
                if power_class == "marginal"
                else "Analysis is underpowered; reduce tau_max, reduce variables, or use longer time series"
            ),
        }

    def _compute_t_eff_single(self, series: np.ndarray) -> float:
        """Compute effective sample size for a single series via integral timescale."""
        n = len(series)
        max_lag = min(n // 4, 200)

        # Compute autocorrelation
        mean = np.mean(series)
        var = np.var(series)
        if var < 1e-10:
            return float(n)

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        for lag in range(1, max_lag + 1):
            cov = np.mean((series[lag:] - mean) * (series[:-lag] - mean))
            acf[lag] = cov / var
            if acf[lag] < 0:
                break

        # Integral timescale
        integral_tau = 1.0 + 2.0 * np.sum(acf[1:])
        t_eff = n / max(integral_tau, 1.0)
        return max(1.0, t_eff)

    def check_confounding_proxies(
        self, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check confounding proxies via environmental invariance.

        Primary: Seasonal/regime Chow tests + residual invariance.
        Auxiliary: VIF (design-matrix conditioning), computed on deseasonalized
        data when seasonality is detected to avoid spurious inflation from
        shared periodic components.
        """
        results = {}

        # Environmental invariance (Chow-style parameter stability)
        if metadata and "seasonal_labels" in metadata:
            results["chow_tests"] = self._test_parameter_stability(
                df, metadata["seasonal_labels"]
            )
        else:
            # Default: split by first/second half
            n = len(df)
            regime_labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
            results["chow_tests"] = self._test_parameter_stability(df, regime_labels)

        # VIF — deseasonalize first if seasonality is present to avoid
        # spurious multicollinearity from shared periodic components
        if len(df.columns) > 1:
            seasonality_check = self.check_seasonality(df)
            seasonality_detected = seasonality_check.get("global", {}).get(
                "seasonality_detected", False
            )
            results["vif"] = self._compute_vif(df, deseasonalize=seasonality_detected)
            results["vif_deseasonalized"] = seasonality_detected

        return results

    def _test_parameter_stability(
        self, df: pd.DataFrame, regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test parameter stability across regimes (Chow-style).

        Returns F-statistics and p-values.
        """
        from scipy.stats import f

        results = {}
        unique_regimes = np.unique(regime_labels)

        if len(unique_regimes) < 2:
            return {"error": "insufficient_regimes"}

        # For each variable, test if AR coefficients differ across regimes
        for col in df.columns:
            # Keep track of which indices are not NaN
            not_nan_mask = ~df[col].isna()
            series = df[col].dropna().values

            if len(series) < 50:
                continue

            # Get regime labels only for non-NaN values
            regime_labels_clean = regime_labels[not_nan_mask.values]

            if len(regime_labels_clean) != len(series):
                # Skip if length mismatch (shouldn't happen but safety check)
                continue

            # Fit AR(2) in each regime
            p = 2
            rss_pooled = self._fit_ar_get_rss(series, p)

            rss_split = 0.0
            k = 0

            for regime in unique_regimes:
                regime_mask = regime_labels_clean == regime
                regime_series = series[regime_mask]

                if len(regime_series) > p + 10:
                    rss_split += self._fit_ar_get_rss(regime_series, p)
                    k += 1

            # Chow F-statistic
            if k >= 2 and rss_split > 0:
                n = len(series)
                F_stat = ((rss_pooled - rss_split) / (p * k)) / (
                    rss_split / (n - k * (p + 1))
                )
                pval = 1 - f.cdf(F_stat, p * k, n - k * (p + 1))

                results[col] = {
                    "chow_f_statistic": float(F_stat),
                    "pvalue": float(pval),
                    "log_p_zscore": -np.log10(pval + 1e-10),
                }

        return results

    def _fit_ar_get_rss(self, series: np.ndarray, p: int) -> float:
        """Fit AR(p) model and return residual sum of squares."""
        X, y = self._prepare_ar_features(series, p)

        if len(X) < p + 5:
            return 0.0

        lr = LinearRegression().fit(X, y)
        pred = lr.predict(X)
        rss = np.sum((y - pred) ** 2)

        return rss

    def _deseasonalize_for_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove trend and seasonal components before VIF computation.

        Seasonal multicollinearity (shared periodic components) inflates VIF
        spuriously. Detrending removes this artifact so VIF reflects genuine
        structural confounding rather than shared seasonality.

        Uses a simple rolling-mean subtraction (window = min(365, T//3)) which
        is fast and does not require knowledge of the seasonal period.
        """
        df_detrended = df.copy()
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 50:
                continue
            window = min(365, max(7, len(series) // 3))
            if window % 2 == 0:
                window += 1
            seasonal = series.rolling(window=window, center=True, min_periods=1).mean()
            df_detrended[col] = df[col] - seasonal
        return df_detrended.dropna()

    def _compute_vif(
        self, df: pd.DataFrame, deseasonalize: bool = False
    ) -> Dict[str, float]:
        """
        Compute Variance Inflation Factor (auxiliary metric).

        When deseasonalize=True, removes trend+seasonal components first to
        avoid spurious VIF inflation from shared periodic components.

        Handles singular/near-singular matrices gracefully by detecting
        extreme multicollinearity and returning appropriate sentinel values.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_results = {}
        df_clean = df.dropna()

        if deseasonalize and len(df_clean) >= 50:
            df_clean = self._deseasonalize_for_vif(df_clean)

        if len(df_clean) < 20 or len(df_clean.columns) < 2:
            return {"error": "insufficient_data"}

        try:
            # Check condition number first (early warning for near-singularity)
            from numpy.linalg import cond

            condition_number = cond(df_clean.values)

            if condition_number > 1e10:
                # Matrix is effectively singular - extreme multicollinearity
                # Return maximal VIF for all variables as a sentinel
                vif_results = {col: float("inf") for col in df_clean.columns}
                vif_results["_condition_number"] = float(condition_number)
                vif_results["_note"] = (
                    "Near-singular matrix: extreme multicollinearity detected"
                )
                return vif_results

            # Attempt VIF calculation per variable
            for i, col in enumerate(df_clean.columns):
                try:
                    vif = variance_inflation_factor(df_clean.values, i)

                    # Handle inf/nan from VIF calculation
                    if np.isnan(vif):
                        vif_results[col] = float("inf")  # Treat NaN as infinite VIF
                    elif np.isinf(vif):
                        vif_results[col] = float("inf")
                    else:
                        vif_results[col] = float(vif)

                except (ValueError, np.linalg.LinAlgError):
                    # Singular matrix for this variable - perfect collinearity
                    vif_results[col] = float("inf")

        except Exception as e:
            # Fallback: return error with condition number if available
            try:
                condition_number = cond(df_clean.values)
                vif_results = {
                    "error": str(e),
                    "_condition_number": float(condition_number),
                    "_note": "VIF calculation failed but matrix appears singular",
                }
            except:
                vif_results = {"error": str(e)}

        return vif_results

    def compute_safe_tau_max(
        self, df: pd.DataFrame, diagnostics: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Compute safe τ_max: both data-driven and budget-driven.

        Returns dictionary with both estimates.
        """
        safe_tau_max = {}

        # Data-driven: ACF zero-crossing / Ljung-Box
        data_driven_tau = []

        for col in df.columns:
            series = df[col].dropna().values

            if len(series) < 50:
                continue

            # Find ACF zero-crossing
            from statsmodels.tsa.stattools import acf

            acf_values = acf(series, nlags=min(50, len(series) // 4), fft=True)

            # First lag where ACF < threshold
            threshold = 1.96 / np.sqrt(len(series))
            zero_crossing = 50  # Default

            for lag in range(1, len(acf_values)):
                if abs(acf_values[lag]) < threshold:
                    zero_crossing = lag
                    break

            data_driven_tau.append(zero_crossing)

        safe_tau_max["data_driven"] = (
            int(np.median(data_driven_tau)) if data_driven_tau else 10
        )

        # Budget-driven: simplified (assume 1000 CI queries/sec, scale with compute budget)
        if self.compute_budget_seconds:
            # Heuristic: tau_max ~ sqrt(budget)
            safe_tau_max["budget_driven"] = int(
                np.sqrt(self.compute_budget_seconds / 10)
            )
        else:
            safe_tau_max["budget_driven"] = None

        return safe_tau_max
