"""
causal-audit: A Framework for Risk Assessment of Assumption Violations
in Time-Series Causal Discovery.

Ruiz et al. (2026). Journal of Causal Inference.
https://github.com/marcoruizrueda/causal-audit
"""

__version__ = "0.2.0"
__author__ = "Marco Ruiz"
__license__ = "MIT"

# Main public API
from .gatekeeper import RiskAwareGatekeeper

# Core modules (can be used independently)
from .a_auditor import AssumptionAuditor
from .b_quantifier import RiskQuantifier
from .c_recommender import MethodRecommender

__all__ = [
    "RiskAwareGatekeeper",
    "AssumptionAuditor",
    "RiskQuantifier",
    "MethodRecommender",
]
