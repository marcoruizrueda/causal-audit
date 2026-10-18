<div align="center">

# Causal-Audit

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 27/27](https://img.shields.io/badge/tests-27%2F27-brightgreen.svg)]()

<p align="center">
  <a href="https://arxiv.org/abs/2604.02488"><b>📄 Paper (arXiv:2604.02488)</b></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/marcoruizrueda/causal-audit/main/framework_overview.svg" width="95%" alt="Framework overview (Figure 1). Stage I computes diagnostics across five assumption families. Stages II–III add calibrated risk scores and an abstention-aware decision policy.">
</p>

A framework for risk assessment of assumption violations in time-series causal discovery.

</div>

> Ruiz, M.¹, Arana-Catania, M.², Ardila, D. R.³ & Ventura, R.¹ (2026).
> *Journal of Causal Inference.*
>
> ¹ ISR, Instituto Superior Técnico, Lisbon &nbsp;
> ² University of Oxford &nbsp;
> ³ Jet Propulsion Laboratory, Caltech

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Framework, figures & benchmarks](#framework-figures--benchmarks)
- [Reproducing paper experiments](#reproducing-paper-experiments)
- [Citation](#citation)
- [License](#license)

## Install

### Users

Install the released package with `uv` (recommended) or `pip`:

```bash
uv pip install causal-audit
# or
pip install causal-audit
```

Optional extras pull in the figure and graph dependencies:

```bash
uv pip install "causal-audit[viz]"     # IEEE-style figures (SciencePlots)
uv pip install "causal-audit[graphs]"  # optional exploratory PCMCI+ graph via tigramite
uv pip install "causal-audit[all]"     # viz + graphs + earth-observation helpers
```

The package installs as `causal-audit` and imports as `causal_audit`.

### Contributors

Clone the repository and install an editable copy into a fresh environment:

```bash
git clone https://github.com/marcoruizrueda/causal-audit.git
cd causal-audit
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

## Quickstart

```python
import pandas as pd
from causal_audit import RiskAwareGatekeeper

# DataFrame with a DatetimeIndex and one column per variable
df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

gk = RiskAwareGatekeeper(random_seed=42)
result = gk.analyze(data=df, output_dir="results/")
print(result["policy"].decision)  # "recommend" or "abstain"
print(result["policy"].recommended_method)  # e.g. "PCMCI+" or "LPCMCI"
```

Run the [quickstart notebook](https://github.com/marcoruizrueda/causal-audit/blob/main/examples/quickstart.ipynb) for a complete audit and its
summary figure. Use the independent [walkthrough](https://github.com/marcoruizrueda/causal-audit/blob/main/examples/walkthrough.ipynb) to
configure the audit, inspect diagnostics, and handle abstention. See [Usage](#usage)
for the per-stage API.

The default audit does not run causal discovery. The optional exploratory graph requires
`include_dependency_network=True` and Tigramite; it is skipped when the audit abstains.

## Usage

Input: a `pandas.DataFrame` with a `DatetimeIndex`, one column per variable, `NaN` for missing values.

```python
import pandas as pd
from causal_audit import AssumptionAuditor  # Stage I
from causal_audit import RiskQuantifier  # Stage II
from causal_audit import MethodRecommender  # Stage III
from causal_audit import RiskAwareGatekeeper  # Stages I → II → III

df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# Stage I — diagnostics (stationarity, irregularity, persistence, nonlinearity, confounding)
auditor = AssumptionAuditor(alpha=0.05)
evidence = auditor.audit(df)  # → AuditEvidence with per-variable effect sizes

# Stage II — diagnostic-derived risk scores with approximate uncertainty bounds
quantifier = RiskQuantifier()
risk_profile = quantifier.quantify(evidence)  # → RiskProfile (6 scores, bounds, ledger)

# Stage III — recommend a method or abstain
recommender = MethodRecommender()
policy, scorecard = recommender.recommend(risk_profile)  # → Policy + Scorecard

# Or run all three stages at once:
gk = RiskAwareGatekeeper(random_seed=42)
result = gk.analyze(data=df, output_dir="results/")
print(result["policy"].decision)  # "recommend" or "abstain"
print(result["policy"].recommended_method)  # "PCMCI+" or "LPCMCI"
```

Stage I can be used alone for structured assumption auditing in individual studies.
Use the gatekeeper for the complete configured pipeline: per-stage defaults do not
automatically load its calibration configuration or retain all downstream diagnostic inputs.

The paper evaluates four calibrated risk dimensions. The current implementation returns
six scores and includes heuristic fallbacks and overrides. Its uncertainty bounds and policy
confidence are not probabilities that a causal graph is correct; low confounding proxies
cannot establish causal sufficiency. Investigate severe nonstationarity and re-audit any
justified preprocessing before selecting a discovery method.

## Framework, figures & benchmarks

The pipeline figure, the figure catalogue (six diagnostic figures plus an optional exploratory graph), the synthetic DGP atlas (F1–F10), the framework extension points, and the nonlinearity-diagnostic notes live in [FRAMEWORK.md](https://github.com/marcoruizrueda/causal-audit/blob/main/causal_audit/FRAMEWORK.md).

## Reproducing paper experiments

See the step-by-step commands in [FRAMEWORK.md](https://github.com/marcoruizrueda/causal-audit/blob/main/causal_audit/FRAMEWORK.md#reproducing-paper-experiments).

## Citation

```bibtex
@misc{ruiz2026causalaudit,
  title   = {Causal-Audit: A Framework for Risk Assessment of Assumption
             Violations in Time-Series Causal Discovery},
  author  = {Marco Ruiz and Miguel Arana-Catania and David R. Ardila
             and Rodrigo Ventura},
  year    = {2026},
  eprint  = {2604.02488},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi     = {10.48550/arXiv.2604.02488},
  url     = {https://arxiv.org/abs/2604.02488},
}
```

MIT License
