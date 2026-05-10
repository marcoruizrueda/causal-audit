"""
Standalone usage example for causal-audit v0.1

This script demonstrates the basic workflow:
1. Generate synthetic data
2. Run gatekeeper analysis
3. Inspect outputs
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import causal-audit
import sys
sys.path.insert(0, '..')
from causal_audit import RiskAwareGatekeeper


def generate_synthetic_data(n=500, seed=42):
    """Generate simple VAR(2) time series for testing."""
    np.random.seed(seed)
    
    # Simple VAR(2) with 3 variables
    n_vars = 3
    data = np.zeros((n, n_vars))
    
    # AR coefficients
    A1 = np.array([[0.7, 0.1, 0.0],
                   [0.2, 0.6, 0.1],
                   [0.0, 0.2, 0.5]])
    
    A2 = np.array([[0.2, 0.0, 0.0],
                   [0.0, 0.1, 0.0],
                   [0.1, 0.0, 0.2]])
    
    # Generate data
    for t in range(2, n):
        data[t] = (A1 @ data[t-1] + 
                   A2 @ data[t-2] + 
                   np.random.normal(0, 0.5, n_vars))
    
    # Create DataFrame with datetime index
    dates = [datetime.now() + timedelta(days=i) for i in range(n)]
    df = pd.DataFrame(
        data[2:],  # Skip first 2 rows (initialization)
        index=pd.DatetimeIndex(dates[2:]),
        columns=['X', 'Y', 'Z']
    )
    
    return df


def main():
    """Run complete example."""
    print("=== causal-audit v0.1 Example ===\n")
    
    # 1. Generate data
    print("1. Generating synthetic VAR(2) data...")
    df = generate_synthetic_data(n=500, seed=42)
    print(f"   Data shape: {df.shape}")
    print(f"   Variables: {list(df.columns)}\n")
    
    # 2. Initialize gatekeeper
    print("2. Initializing RiskAwareGatekeeper...")
    gatekeeper = RiskAwareGatekeeper(random_seed=42)
    print("   ✓ Gatekeeper initialized\n")
    
    # 3. Run analysis
    print("3. Running analysis...")
    results = gatekeeper.analyze(
        data=df,
        metadata=None,
        compute_budget=None,
        output_dir="test_output/"
    )
    print("   ✓ Analysis complete\n")
    
    # 4. Inspect results
    print("4. Results Summary:")
    print(f"   Decision: {results['policy'].decision.upper()}")
    
    if results['policy'].decision == "recommend":
        print(f"   Recommended Method: {results['policy'].recommended_method}")
        print(f"   Confidence: {results['policy'].confidence:.2%}")
    else:
        print("   Reason: Data quality insufficient")
    
    print(f"\n5. Risk Profile:")
    for risk_name, risk_vals in results['risk_profile'].risks.items():
        mean = risk_vals['mean']
        lower = risk_vals['lower_95']
        upper = risk_vals['upper_95']
        print(f"   {risk_name}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
    
    print(f"\n6. Outputs saved to: test_output/")
    print("   - audit_evidence.json")
    print("   - risk_profile.json")
    print("   - recommendation_policy.json")
    print("   - audit_scorecard.md (read this!)")
    
    print("\n=== Example Complete ===")
    print("Next: Read test_output/audit_scorecard.md for full report")


if __name__ == "__main__":
    main()

