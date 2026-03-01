"""
model_validation.py
───────────────────
Quality gate for CI/CD pipeline.

Usage:
    python model_validation.py                      # auto-picks best run
    python model_validation.py <path/to/run.json>   # validate specific run

Exit codes:
    0 = PASSED  (pipeline continues)
    1 = FAILED  (pipeline stops)
"""

import sys
import json
import os
import glob

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_R2_SCORE = 0.40   # model must explain at least 40 % of variance
MAX_RMSE     = 65.0   # root-mean-square-error ceiling (diabetes target scale)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_run(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def find_best_run() -> tuple[dict, str]:
    """Return (run_data, filepath) for the run with the highest R² score."""
    candidates = glob.glob("tracking_evidence/run_*.json")

    if not candidates:
        print("FAILED: No run files found in tracking_evidence/")
        sys.exit(1)

    best_r2, best_data, best_path = -float("inf"), None, None
    for path in candidates:
        try:
            data = load_run(path)
            r2 = data["metrics"]["r2_score"]
            if r2 > best_r2:
                best_r2, best_data, best_path = r2, data, path
        except Exception as e:
            print(f"  WARNING: Could not read {path}: {e}")

    if best_data is None:
        print("FAILED: No valid run files found.")
        sys.exit(1)

    return best_data, best_path


# ── Main validation ───────────────────────────────────────────────────────────
def validate_model():
    # Resolve which run to validate
    if len(sys.argv) > 1:
        evidence_path = sys.argv[1]
        if not os.path.exists(evidence_path):
            print(f"FAILED: File not found → {evidence_path}")
            sys.exit(1)
        run_data = load_run(evidence_path)
        print(f"Validating specified run: {evidence_path}")
    else:
        run_data, evidence_path = find_best_run()
        print(f"Auto-selected best run: {evidence_path}")

    # ── Extract values ────────────────────────────────────────────────────────
    try:
        metrics = run_data["metrics"]
        r2   = metrics["r2_score"]
        rmse = metrics.get("rmse", None)
        tags = run_data.get("tags", {})
        model_hash = tags.get("model_hash", None)
        params = run_data.get("params", {})
    except KeyError as e:
        print(f"FAILED: Malformed run data — missing key {e}")
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Run Summary ──────────────────────────────────────────────────")
    print(f"  Params      : {params}")
    print(f"  R² Score    : {r2:.4f}  (threshold ≥ {MIN_R2_SCORE})")
    if rmse is not None:
        print(f"  RMSE        : {rmse:.4f}  (threshold ≤ {MAX_RMSE})")
    if model_hash:
        print(f"  Model Hash  : {model_hash[:32]}...")
    else:
        print("  Model Hash  : ⚠️  NOT FOUND — artifact integrity unverified")
    print("─────────────────────────────────────────────────────────────────\n")

    # ── Gate 1: R² threshold ──────────────────────────────────────────────────
    if r2 < MIN_R2_SCORE:
        print(f"❌ FAILED — R² {r2:.4f} is below minimum {MIN_R2_SCORE}")
        sys.exit(1)

    # ── Gate 2: RMSE ceiling ──────────────────────────────────────────────────
    if rmse is not None and rmse > MAX_RMSE:
        print(f"❌ FAILED — RMSE {rmse:.4f} exceeds maximum {MAX_RMSE}")
        sys.exit(1)

    # ── Gate 3: Artifact hash present ────────────────────────────────────────
    if not model_hash:
        print("❌ FAILED — model_hash tag is missing; cannot verify artifact integrity")
        sys.exit(1)

    # ── All gates passed ──────────────────────────────────────────────────────
    print("✅ PASSED — all quality gates met. Model approved for staging.")
    sys.exit(0)


if __name__ == "__main__":
    validate_model()