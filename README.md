# Causal-Audit

A framework for risk assessment of assumption violations in time-series causal discovery.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 27/27](https://img.shields.io/badge/tests-27%2F27-brightgreen.svg)]()

> Ruiz, M.¹, Arana-Catania, M.², Ardila, D. R.³ & Ventura, R.¹ (2026).
> *Journal of Causal Inference.*
>
> ¹ ISR, Instituto Superior Técnico, Lisbon &nbsp;
> ² University of Oxford &nbsp;
> ³ Jet Propulsion Laboratory, Caltech

<p align="center">
  <img src="framework_overview.svg" width="95%" alt="Framework overview (Figure 1). Stage I computes diagnostics across five assumption families. Stages II–III add calibrated risk scores and an abstention-aware decision policy.">
</p>

## Installation

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

## Usage

Input: a `pandas.DataFrame` with a `DatetimeIndex`, one column per variable, `NaN` for missing values.

```python
import pandas as pd
from causal_audit import AssumptionAuditor   # Stage I
from causal_audit import RiskQuantifier      # Stage II
from causal_audit import MethodRecommender   # Stage III
from causal_audit import RiskAwareGatekeeper # Stages I → II → III

df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# Stage I — diagnostics (stationarity, irregularity, persistence, nonlinearity, confounding)
auditor = AssumptionAuditor(alpha=0.05)
evidence = auditor.audit(df)               # → AuditEvidence with per-variable effect sizes

# Stage II — calibrated risk scores with 95 % credible intervals
quantifier = RiskQuantifier()
risk_profile = quantifier.quantify(evidence) # → RiskProfile (4 risks, CrI, ledger)

# Stage III — recommend a method or abstain
recommender = MethodRecommender()
policy, scorecard = recommender.recommend(risk_profile) # → Policy + Scorecard

# Or run all three stages at once:
gk = RiskAwareGatekeeper(random_seed=42)
result = gk.analyze(data=df, output_dir="results/")
print(result["policy"].decision)           # "recommend" or "abstain"
print(result["policy"].recommended_method) # "PCMCI+" or "Granger"
```

Stage I can be used alone for structured assumption auditing in individual studies. The full pipeline adds calibrated risk estimation and decision support.

## Synthetic DGP Atlas (Zenodo)

500 synthetic multivariate time series with controlled assumption violations, designed for benchmarking causal discovery pre-processing and method selection. All families share a VAR(1) base process. Generated 2025-11-30, seed 42.

| Family | Name | n | N | T | Violation mechanism | Severity |
|--------|------|---|---|---|---------------------|----------|
| F1 | Clean baseline | 50 | 5–8 | 500–1000 | None | ρ(A) ≤ 0.7 |
| F2 | Structural breaks | 50 | 5–8 | 500–1000 | 1–3 regime changes in VAR coefficients | Continuous |
| F3 | Irregular sampling | 50 | 5–8 | 500–1000 | MCAR / MAR / seasonal gaps | 15–35 % missing |
| F4 | High persistence | 50 | 5–8 | 500–1000 | Near-unit-root spectral radius | ρ(A) ∈ [0.92, 0.98] |
| F5 | Latent confounders | 50 | 5–8 | 500–1000 | L ∈ {1, 2} hidden common causes | σ_conf ∈ {0.3, 0.6, 0.9} |
| F6 | Seasonality | 50 | 5–8 | 500–1000 | Additive harmonic components | P ∈ {12, 24, 52} |
| F7 | Nonlinear | 50 | 5–8 | 500–1000 | tanh / sin / ReLU transforms | Moderate |
| F8 | Non-Gaussian | 50 | 5–8 | 500–1000 | Student-t or Laplace noise | ν ∈ {3, 5, 10} |
| F9 | Mixed violations | 50 | 5–8 | 500–1000 | 2–3 families combined | Multiple high |
| F10 | Boundary cases | 50 | 3–12 | 200–2000 | Short series / high-dim / sparse / near-unit-root | Stress test |

Generator: `causal_audit_validation/generators/synthetic_atlas_extended_v02.py`.

## Extending

Each stage has a single extension point.

**New method.** Add to `config/method_catalog.yaml`:

```yaml
LPCMCI:
  full_name: "Latent PCMCI"
  citation: "Gerhardus & Runge (2020)"
  implementation: "tigramite"
  assumptions:
    stationarity: { tolerance: "soft", penalty_weight: 0.5 }
    causal_sufficiency: { tolerance: "none" }   # handles latent confounders
  parameters: { tau_max: "data_driven", pc_alpha: 0.01 }
```

Then register its risk thresholds in `c_recommender.py`:

```python
# In METHOD_CONSTRAINTS:
"LPCMCI": {
    "hard_constraints": [
        ("NonstationarityRisk", 0.80, "Requires weak stationarity"),
    ],
    "soft_constraints": [
        ("ConfoundingRisk", 0.90, "Handles latent confounders via PAGs"),
        ("PersistenceRisk", 0.85, "Large tau_max needed"),
    ],
    "base_cost": 2.0,
}
```

**New diagnostic.** Add a method to `a_auditor.py` and wire it in `audit()`:

```python
# In AssumptionAuditor:
def check_measurement_noise(self, df):
    results = {}
    for col in df.columns:
        series = df[col].dropna().values
        # e.g. ratio of high-frequency variance to total variance
        results[col] = {"hf_variance_ratio": float(...)}
    return results

# In audit():
diagnostics["measurement_noise"] = self.check_measurement_noise(df)
```

**New risk dimension.** Three edits: append to `CORE_RISKS` in `b_quantifier.py`, add extraction in `_extract_diagnostic_values()`, and add weights in `calib_v2.yaml`:

```yaml
# In global_parameters.risks:
MeasurementNoiseRisk:
  alpha: -2.0
  diagnostic_weights:
    hf_variance_ratio: { weight: 3.0 }
```

**Recalibrate.** Run your DGPs through Stage I, fit new models, freeze to YAML, and point the gatekeeper to it: `RiskAwareGatekeeper(calib="calib_v3.yaml")`. Scripts: `causal_audit_validation/calibration_v02/`.

See `docs/threat_model.md` for what the framework cannot detect.

## Reproducing paper experiments

```bash
# 1. Synthetic Atlas: generate 500 DGPs, run calibration, cross-validate
python causal_audit_validation/generators/synthetic_atlas_extended_v02.py
python causal_audit_validation/calibration_v02/run_full_atlas_calibration_v02.py
python causal_audit_validation/calibration_v02/run_cross_validation.py

# 2. TimeGraph benchmark (18 categories)
python timegraph_validation/run_validation_v3_final.py

# 3. CausalTime benchmark (3 domains)
python causaltime_validation/run_causaltime_validation.py
```

## Citation

```bibtex
@article{ruiz2026causal_audit,
  title   = {Causal-Audit: A Framework for Risk Assessment of Assumption
             Violations in Time-Series Causal Discovery},
  author  = {Ruiz, Marco and Arana-Catania, Miguel and Ardila, David R.
             and Ventura, Rodrigo},
  journal = {Journal of Causal Inference},
  year    = {2026},
  url     = {https://github.com/marcoruizrueda/causal-audit}
}
```

MIT License
