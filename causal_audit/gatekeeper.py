"""
Main API: RiskAwareGatekeeper

Orchestrates the complete audit → quantify → recommend pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

from .a_auditor import AssumptionAuditor, AuditEvidence
from .b_quantifier import RiskQuantifier, RiskProfile
from .c_recommender import MethodRecommender, Policy, Scorecard
from .utils.io import (
    save_json,
    save_markdown,
    ensure_dir,
    check_output_directory_writable,
)
from .utils.schema_validation import (
    validate_audit_evidence,
    validate_risk_profile,
    validate_policy,
)


class RiskAwareGatekeeper:
    """
    Main user-facing API for causal-audit package.

    Orchestrates: Audit (A) → Quantify (B) → Recommend (C).
    """

    def __init__(
        self,
        method_catalog: str = "config/method_catalog.yaml",
        priors: str = "config/priors_v1.yaml",
        calib: Optional[str] = "config/calib_v2.yaml",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize gatekeeper.

        Args:
            method_catalog: Path to method_catalog.yaml
            priors: Path to priors_v1.yaml
            calib: Path to calib_v1.yaml (optional)
            random_seed: Random seed for reproducibility
        """
        # Resolve paths relative to package
        package_dir = Path(__file__).parent

        self.method_catalog_path = self._resolve_path(method_catalog, package_dir)
        self.priors_path = self._resolve_path(priors, package_dir)
        self.calib_path = self._resolve_path(calib, package_dir) if calib else None

        self.random_seed = random_seed

        # Initialize modules
        self.auditor = AssumptionAuditor(alpha=0.05, random_seed=random_seed)

        self.quantifier = RiskQuantifier(
            calibration_file=self.calib_path, random_seed=random_seed
        )

        self.recommender = MethodRecommender(
            method_catalog_file=self.method_catalog_path,
            priors_file=self.priors_path,
            calibration_file=self.calib_path,
        )

    def _resolve_path(self, path: str, package_dir: Path) -> str:
        """Resolve config path relative to package directory."""
        path_obj = Path(path)

        if path_obj.is_absolute():
            return str(path_obj)

        # Try relative to package
        package_relative = package_dir / path
        if package_relative.exists():
            return str(package_relative)

        # Try relative to current directory
        if path_obj.exists():
            return str(path_obj)

        # Return as-is and let downstream handle error
        return path

    def analyze(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        compute_budget: Optional[float] = None,
        output_dir: str = "results/",
        force_proceed: bool = False,
        missing_value_codes: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.

        Args:
            data: Time-series DataFrame (rows=time, columns=variables)
            metadata: Optional metadata (seasonal_labels, qc_flags, etc.)
            compute_budget: Optional compute budget in seconds
            output_dir: Output directory for all artifacts
            force_proceed: If True, override abstention and recommend best method
                          with explicit CRITICAL warnings (use with extreme caution!)
            missing_value_codes: Optional list of values to treat as missing (e.g., [-9999, -999])
                                Common codes: FLUXNET=-9999, NASA POWER=-999, WMO=9999

        Returns:
            Dictionary with all outputs (audit_evidence, risk_profile, policy)
        """
        # Validate output directory
        if not check_output_directory_writable(output_dir):
            raise IOError(f"Output directory {output_dir} is not writable")

        # Create output subdirectories
        figures_dir = Path(output_dir) / "figures"
        tables_dir = Path(output_dir) / "diagnostic_tables"
        ensure_dir(figures_dir)
        ensure_dir(tables_dir)

        # Preprocess: Handle domain-specific missing value codes
        if missing_value_codes is not None:
            print(f"  Converting missing value codes to NaN: {missing_value_codes}")
            data = data.replace(missing_value_codes, np.nan)

        # Phase 1: Audit (Module A)
        print("Phase 1/3: Running assumption audit...")
        audit_evidence = self.auditor.audit(data, metadata)

        # Validate and save
        validate_audit_evidence(audit_evidence.to_dict())
        save_json(audit_evidence.to_dict(), Path(output_dir) / "audit_evidence.json")

        # Generate audit summary (markdown)
        audit_summary = self._generate_audit_summary(audit_evidence)
        save_markdown(audit_summary, Path(output_dir) / "audit_summary.md")

        # Save diagnostic tables
        self._save_diagnostic_tables(audit_evidence, tables_dir)

        # Phase 2: Quantify (Module B)
        print("Phase 2/3: Quantifying risks...")
        risk_profile = self.quantifier.quantify(audit_evidence)

        # Validate and save
        validate_risk_profile(risk_profile.to_dict())
        save_json(risk_profile.to_dict(), Path(output_dir) / "risk_profile.json")

        # Generate risk summary (markdown)
        risk_summary = self._generate_risk_summary(risk_profile)
        save_markdown(risk_summary, Path(output_dir) / "risk_summary.md")

        # Phase 3: Recommend (Module C)
        print("Phase 3/3: Generating recommendation...")
        if force_proceed:
            print(
                "  ⚠️  WARNING: force_proceed=True - overriding abstention with explicit warnings!"
            )
        policy, scorecard = self.recommender.recommend(
            risk_profile,
            compute_budget,
            t_eff_ratio=float(
                np.mean(list(audit_evidence.t_eff.values()))
                / max(1, audit_evidence.n_samples)
            )
            if audit_evidence.t_eff
            else None,
            force_proceed=force_proceed,
        )

        # Validate and save
        validate_policy(policy.to_dict())
        save_json(policy.to_dict(), Path(output_dir) / "recommendation_policy.json")

        # Save scorecard (markdown)
        scorecard_md = scorecard.to_markdown()
        save_markdown(scorecard_md, Path(output_dir) / "audit_scorecard.md")

        # Generate plots (all IEEE style, single consolidated call)
        try:
            from .plotting import generate_all_figures

            print("Generating figures...")
            generate_all_figures(
                data=data,
                audit_evidence_dict=audit_evidence.to_dict(),
                risk_profile_dict=risk_profile.to_dict(),
                policy_dict=policy.to_dict(),
                output_dir=Path(output_dir),
            )
            print("  ✓ Generated 7 figures (IEEE style, 300 DPI)")

        except Exception as e:
            print(f"  ⚠ Could not generate figures: {e}")
            print(
                f"  (Install requirements: pip install matplotlib SciencePlots scipy statsmodels)"
            )

        print(f"\n✓ Analysis complete! Results saved to {output_dir}")
        print(f"  - Decision: {policy.decision.upper()}")
        if policy.recommended_method:
            print(f"  - Recommended method: {policy.recommended_method}")
            print(f"  - Confidence: {policy.confidence:.2%}")

        return {
            "audit_evidence": audit_evidence,
            "risk_profile": risk_profile,
            "policy": policy,
            "scorecard": scorecard,
            "output_dir": output_dir,
        }

    def _generate_audit_summary(self, evidence: AuditEvidence) -> str:
        """Generate human-readable audit summary."""
        md = "# Assumption Audit Summary\n\n"
        md += f"**Variables**: {evidence.n_variables}\n\n"
        md += f"**Samples**: {evidence.n_samples}\n\n"

        md += "## Diagnostic Summary\n\n"

        # Stationarity
        md += "### Stationarity\n\n"
        for var, stats in evidence.diagnostics.get("stationarity", {}).items():
            if "error" not in stats:
                md += f"- **{var}**: Break magnitude = {stats.get('break_magnitude_sigma', 0):.2f}σ\n"

        md += "\n### Irregularity\n\n"
        gap_cv = (
            evidence.diagnostics.get("irregularity", {}).get("global", {}).get("gap_cv")
        )
        if gap_cv:
            md += f"- Gap coefficient of variation: {gap_cv:.3f}\n"

        md += "\n### Persistence\n\n"
        for var, t_eff_val in evidence.t_eff.items():
            md += f"- **{var}**: T_eff = {t_eff_val:.0f}\n"

        md += f"\n### Safe τ_max\n\n"
        md += f"- Data-driven: {evidence.safe_tau_max.get('data_driven', 'N/A')}\n"
        md += f"- Budget-driven: {evidence.safe_tau_max.get('budget_driven', 'N/A')}\n"

        return md

    def _generate_risk_summary(self, profile: RiskProfile) -> str:
        """Generate human-readable risk summary."""
        md = "# Risk Profile Summary\n\n"

        md += "## Composite Risks\n\n"

        for risk_name in profile.risk_names:
            risk_vals = profile.risks[risk_name]
            mean = risk_vals["mean"]
            lower = risk_vals["lower_95"]
            upper = risk_vals["upper_95"]

            md += f"### {risk_name}\n\n"
            md += f"- **Mean**: {mean:.3f}\n"
            md += f"- **95% CrI**: [{lower:.3f}, {upper:.3f}]\n"
            md += f"- **Width**: {upper - lower:.3f}\n"

            # Risk ledger
            md += "\n**Top Contributors**:\n\n"
            for contrib in profile.risk_ledger.get(risk_name, []):
                # Handle both old format (effect_size) and new format (value + contribution)
                if "value" in contrib and "contribution" in contrib:
                    md += f"- {contrib['diagnostic']}: value = {contrib['value']:.3f}, contribution = {contrib['contribution']:.3f}\n"
                elif "effect_size" in contrib:
                    md += f"- {contrib['diagnostic']}: effect = {contrib['effect_size']:.3f}\n"
                else:
                    md += f"- {contrib['diagnostic']}: (details not available)\n"

            md += "\n"

        return md

    def _save_diagnostic_tables(self, evidence: AuditEvidence, tables_dir: Path):
        """Save diagnostic tables as CSV."""
        # Stationarity
        stationarity_rows = []
        for var, stats in evidence.diagnostics.get("stationarity", {}).items():
            if "error" not in stats:
                row = {
                    "variable": var,
                    "break_magnitude_sigma": stats.get("break_magnitude_sigma", 0),
                    "drift_slope": stats.get("drift_slope_normalized", 0),
                    "adf_pvalue": stats.get("adf", {}).get("pvalue", 1.0),
                    "kpss_pvalue": stats.get("kpss", {}).get("pvalue", 1.0),
                }
                stationarity_rows.append(row)

        if stationarity_rows:
            df_stationarity = pd.DataFrame(stationarity_rows)
            df_stationarity.to_csv(tables_dir / "stationarity_tests.csv", index=False)

        # Irregularity
        irregularity_rows = []
        for var, stats in (
            evidence.diagnostics.get("irregularity", {}).get("per_variable", {}).items()
        ):
            row = {
                "variable": var,
                "missing_rate": stats.get("missing_rate", 0),
                "mnar_pvalue": stats.get("mnar_seasonal", {}).get("pvalue", 1.0),
                "appears_mcar": stats.get("appears_mcar", True),
            }
            irregularity_rows.append(row)

        if irregularity_rows:
            df_irregularity = pd.DataFrame(irregularity_rows)
            df_irregularity.to_csv(tables_dir / "irregularity_tests.csv", index=False)

        # Persistence
        persistence_rows = []
        for var, stats in evidence.diagnostics.get("persistence", {}).items():
            if "error" not in stats:
                row = {
                    "variable": var,
                    "t_eff": stats.get("t_eff", 0),
                    "t_eff_ratio": stats.get("t_eff_ratio", 1.0),
                    "integral_tau": stats.get("integral_tau", 0),
                }
                persistence_rows.append(row)

        if persistence_rows:
            df_persistence = pd.DataFrame(persistence_rows)
            df_persistence.to_csv(tables_dir / "persistence_tests.csv", index=False)

        # Confounding
        confounding_rows = []
        for var, stats in (
            evidence.diagnostics.get("confounding", {}).get("chow_tests", {}).items()
        ):
            if isinstance(stats, dict) and "pvalue" in stats:
                row = {
                    "variable": var,
                    "chow_f_statistic": stats.get("chow_f_statistic", 0),
                    "pvalue": stats.get("pvalue", 1.0),
                }
                confounding_rows.append(row)

        if confounding_rows:
            df_confounding = pd.DataFrame(confounding_rows)
            df_confounding.to_csv(tables_dir / "confounding_proxies.csv", index=False)
