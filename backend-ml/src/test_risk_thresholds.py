"""
Risk Stratification Threshold Verification
───────────────────────────────────────────
Validates the corrected risk classification logic against
the percentile-based framework:

    Low      → probability < 0.3451
    Moderate → 0.3451 ≤ probability ≤ 0.9428
    High     → probability > 0.9428

Usage:
    python -m src.test_risk_thresholds
"""

import sys
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.risk_classification import get_risk_level


def run_threshold_tests():
    """Runs boundary and standard test cases for risk classification."""

    # Load thresholds from metadata (same source serve_model.py uses)
    metadata_path = root / "outputs" / "gradient_boosting" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    risk_cfg = metadata.get("risk_thresholds", {})
    q1 = risk_cfg.get("q1", 0.25)
    q3 = risk_cfg.get("q3", 0.75)

    print("=" * 60)
    print("  RISK THRESHOLD VERIFICATION TEST")
    print("=" * 60)
    print(f"\n  Q1 (Low/Moderate boundary):  {q1}")
    print(f"  Q3 (Moderate/High boundary): {q3}\n")

    # Test cases: (probability, expected_risk_level)
    test_cases = [
        # Standard cases from user spec
        (0.20,   "Low"),
        (0.50,   "Moderate"),
        (0.9428, "Moderate"),   # Exact Q3 boundary → Moderate (<=)
        (0.9434, "High"),       # The misclassification case → must be High
        (0.99,   "High"),

        # Edge cases
        (0.0,    "Low"),        # Minimum probability
        (1.0,    "High"),       # Maximum probability
        (0.3451, "Moderate"),   # Exact Q1 boundary → Moderate (>=)
        (0.3450, "Low"),        # Just below Q1 → Low
        (0.9429, "High"),       # Just above Q3 → High

        # Float precision cases
        (0.34509999, "Low"),    # Floating-point just below Q1
        (0.94280001, "High"),   # Floating-point just above Q3
    ]

    all_passed = True
    for prob, expected in test_cases:
        result = get_risk_level(prob, q1, q3)
        actual = result["level"]
        status = "PASS" if actual == expected else "FAIL"

        if status == "FAIL":
            all_passed = False

        # Verify all fields are present and synchronized
        assert "level" in result, f"Missing 'level' in result for prob={prob}"
        assert "color" in result, f"Missing 'color' in result for prob={prob}"
        assert "percentile" in result, f"Missing 'percentile' in result for prob={prob}"
        assert "action" in result, f"Missing 'action' in result for prob={prob}"

        # Verify color/percentile/action match the level
        level_colors = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}
        level_percentiles = {"Low": "Bottom 25%", "Moderate": "Middle 50%", "High": "Top 25%"}

        assert result["color"] == level_colors[actual], \
            f"Color mismatch for {actual}: {result['color']} != {level_colors[actual]}"
        assert result["percentile"] == level_percentiles[actual], \
            f"Percentile mismatch for {actual}: {result['percentile']} != {level_percentiles[actual]}"

        print(f"  [{status}]  prob={prob:<12}  expected={expected:<10}  got={actual:<10}  "
              f"color={result['color']}  percentile={result['percentile']}")

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED [OK]")
    else:
        print("  SOME TESTS FAILED [ERROR]")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_threshold_tests()
    sys.exit(0 if success else 1)
