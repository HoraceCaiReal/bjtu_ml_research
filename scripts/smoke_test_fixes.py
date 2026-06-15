"""
Quick smoke test for BUG-001 through BUG-005 fixes.
Runs only the critical paths that were previously failing.

Usage:
    conda activate bjtu_ml
    python scripts/smoke_test_fixes.py
"""
import sys
import os
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from test_integration (no __init__.py in tests/)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "test_integration", str(PROJECT_ROOT / "tests" / "test_integration.py"),
)
_ti = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ti)
run_single_test = _ti.run_single_test


SMOKE_TESTS = [
    # BUG-001: CNN pretrained (was: state_dict mismatch)
    {"id": "D1_cnn_ce_adam", "model_name": "cnn", "cnn_loss_fn": "cross_entropy",
     "cnn_epochs": 3, "cnn_early_stopping": 2},
    {"id": "D2_cnn_focal_g2", "model_name": "cnn", "cnn_loss_fn": "focal",
     "cnn_focal_gamma": 2.0, "cnn_focal_alpha": "None",
     "cnn_epochs": 3, "cnn_early_stopping": 2},

    # BUG-002: Unsupervised pretrained (was: labels_ dimension mismatch)
    {"id": "E1_kmeans_lloyd", "model_name": "kmeans", "kmeans_algorithm": "lloyd"},
    {"id": "E3_gmm_full", "model_name": "gmm", "gmm_covariance_type": "full"},
    {"id": "E8_agg_ward", "model_name": "agglomerative", "agg_linkage": "ward"},
    {"id": "E10_spec_rbf", "model_name": "spectral", "spec_affinity": "rbf"},

    # BUG-005: DBSCAN (was: all noise)
    {"id": "E7_dbscan_manual", "model_name": "dbscan", "optimization_strategy": "manual"},
]


def main():
    print("=" * 60)
    print("Smoke Test: BUG-001 through BUG-005 Fixes")
    print("=" * 60)

    data_cache = {}
    passed = 0
    total = len(SMOKE_TESTS)
    results = []

    for tc in SMOKE_TESTS:
        test_id = tc["id"]
        print(f"\nRunning {test_id}...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = run_single_test(test_id, tc, data_cache)
            elapsed = time.time() - t0
            ok = result.get("passed", False)
            results.append({"id": test_id, "passed": ok, "elapsed": elapsed})

            if ok:
                passed += 1
                print(f"PASS ({elapsed:.1f}s)")
            else:
                status = result.get("status_preview", "unknown")
                print(f"FAIL ({elapsed:.1f}s)")
                print(f"  -> {status[:150]}")
        except Exception as e:
            elapsed = time.time() - t0
            results.append({"id": test_id, "passed": False, "elapsed": elapsed})
            print(f"ERROR ({elapsed:.1f}s): {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print(f"{'=' * 60}")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['id']} ({r['elapsed']:.1f}s)")

    if passed < total:
        print(f"\n{total - passed} test(s) failed!")
        sys.exit(1)
    else:
        print("\nAll smoke tests passed!")


if __name__ == "__main__":
    main()
