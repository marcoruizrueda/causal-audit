"""Do not turn uncalibrated binned MI into nonlinearity claims."""
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causal_audit.a_auditor import AssumptionAuditor
from causal_audit.b_quantifier import RiskQuantifier
from causal_audit.plotting.figures import _plot_prediscovery_summary


@pytest.mark.parametrize('kind', ['independent', 'linear', 'nonlinear'])
def test_binned_mi_is_descriptive_not_a_nonlinearity_test(kind):
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    y = rng.normal(size=600)
    if kind == 'linear':
        y += 2 * x
    elif kind == 'nonlinear':
        y += 2 * x ** 2
    result = AssumptionAuditor().check_pairwise_nonlinearity(pd.DataFrame({'X': x, 'Y': y}))
    assert result['summary']['status'] == 'descriptive_only'
    assert result['summary']['fraction_nonlinear'] is None
    assert result['summary']['recommended_ci_test'] is None
    pair = result['pairs']['X_Y']
    assert pair['is_nonlinear'] is None
    assert np.isfinite(pair['mi_estimate'])


def test_unavailable_pairwise_fraction_does_not_break_quantification():
    evidence = SimpleNamespace(diagnostics={'pairwise_nonlinearity': {'summary': {'fraction_nonlinear': None}}})
    result = RiskQuantifier()._extract_diagnostic_values(evidence)
    assert result['NonlinearityRisk']['fraction_nonlinear_pairs'] == 0


def test_summary_uses_predictive_diagnostic_and_policy(monkeypatch, tmp_path):
    data = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 2)), columns=['X', 'Y'])
    diagnostics = {'nonlinearity': {'X': {'delta_rmse_relative': -0.05}, 'Y': {'delta_rmse_relative': 0.01}},
                   'pairwise_nonlinearity': {'pairs': {'X_Y': {'r_squared': 0.01, 'mi_estimate': 0.3, 'is_nonlinear': True}},
                                            'summary': {'fraction_nonlinear': 1.0, 'recommended_ci_test': 'cmiknn'}}}
    saved = []
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', lambda fig, *a, **k: saved.append(fig))
    _plot_prediscovery_summary(data, list(data), {'diagnostics': diagnostics}, {'risks': {}},
                              {'recommended_method': 'VARLiNGAM', 'confidence': 0.85}, tmp_path, None)
    fig = saved[0]
    titles = ' '.join(ax.get_title() for ax in fig.axes)
    text = ' '.join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert '100% nonlinear' not in titles
    assert 'Prediction' in titles
    assert 'Use PCMCI+ with ParCorr' not in text
    assert 'Check VARLiNGAM assumptions' in text
    plt.close('all')


from types import SimpleNamespace
from causal_audit.c_recommender import MethodRecommender

def test_low_risk_does_not_establish_nongaussian_innovations():
    names = ['NonstationarityRisk', 'IrregularityRisk', 'PersistenceRisk', 'ConfoundingRisk', 'NonlinearityRisk', 'SeasonalityRisk']
    profile = SimpleNamespace(risks={name: {'mean': .1, 'lower_95': .05, 'upper_95': .15} for name in names}, risk_names=names, risk_ledger={})
    policy, scorecard = MethodRecommender().recommend(profile)
    assert policy.recommended_method == 'PCMCI+'
    assert policy.method_config['pcmci']['test_method'] == 'parcorr'
    assert not policy.method_config.get('varlingam', {}).get('enabled', False)
    assert 'non-Gaussian' in scorecard.rationale


@pytest.mark.parametrize('nonlinearity', [0.1, 0.6])
@pytest.mark.parametrize('seasonality', [0.1, 0.9])
def test_extreme_nonstationarity_is_not_cleared_by_proxy_scores(nonlinearity, seasonality):
    """Neither proxy can establish that preprocessing restores stationarity."""
    names = ['NonstationarityRisk', 'IrregularityRisk', 'PersistenceRisk',
             'ConfoundingRisk', 'NonlinearityRisk', 'SeasonalityRisk']
    risks = {name: {'mean': 0.1, 'lower_95': 0.05, 'upper_95': 0.15} for name in names}
    for name, score in [('NonstationarityRisk', 0.99),
                        ('NonlinearityRisk', nonlinearity),
                        ('SeasonalityRisk', seasonality)]:
        risks[name] = {'mean': score, 'lower_95': max(0, score - 0.05),
                       'upper_95': min(1, score + 0.05)}
    profile = SimpleNamespace(risks=risks, risk_names=names, risk_ledger={})
    policy, scorecard = MethodRecommender().recommend(profile)
    assert policy.decision == 'abstain'
    assert 'CRITICAL_NONSTATIONARITY' in policy.policy_reason_codes
    assert 're-audit' in scorecard.rationale


@pytest.mark.parametrize('opt_in,decision,expected_graphs', [
    (False, 'recommend', 0), (True, 'abstain', 0), (True, 'recommend', 1),
])
def test_exploratory_discovery_requires_opt_in_and_a_recommendation(
        monkeypatch, tmp_path, opt_in, decision, expected_graphs):
    """Audit-only and abstention paths must never call the discovery runner."""
    from causal_audit.plotting import figures
    data = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 2)),
                        index=pd.date_range('2020-01-01', periods=100), columns=['X', 'Y'])
    calls = []
    backend_changes = []
    monkeypatch.setattr(matplotlib, 'use', lambda *args, **kwargs: backend_changes.append(args))

    def exploratory_graph(data, cols, evidence, save):
        calls.append(True)
        save(plt.figure(), 'dependency_network')

    monkeypatch.setattr(figures, '_plot_dependency_network', exploratory_graph)
    policy = {'decision': decision, 'recommended_method': 'PCMCI+' if decision == 'recommend' else None,
              'confidence': 0.85 if decision == 'recommend' else None}
    paths = figures.generate_all_figures(data, {'diagnostics': {}}, {'risks': {}},
                                        policy, tmp_path, include_dependency_network=opt_in)
    assert len(calls) == expected_graphs
    assert backend_changes == []
    assert len(paths) == 6 + expected_graphs
    assert all(path.exists() and path.with_suffix('.pdf').exists() for path in paths)
