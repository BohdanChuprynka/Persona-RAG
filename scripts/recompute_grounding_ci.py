"""Add question-clustered bootstrap CIs to a grounding-probe results.json (review fix #7).

The probe decodes each of 30 identity questions K=5 times per condition, so the
150 generations per condition are NOT independent — the independent unit is the
question. Wilson intervals on n=150 therefore overstate precision. This reads the
per-probe label lists already saved in ``results.json`` and writes a ``cluster_ci``
block (correct / hallucinated / deflected, each condition + the bare→grounded
delta) computed by ``cluster_bootstrap_rate_ci``. Pure recompute — no model/API.

    uv run python scripts/recompute_grounding_ci.py reports/main/grounding/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from persona_rag.eval.compare import cluster_bootstrap_rate_ci

LABELS = ("correct", "hallucinated", "deflected")


def recompute(path: Path, *, n_boot: int = 10000, seed: int = 0) -> dict:
    res = json.loads(path.read_text(encoding="utf-8"))
    per_probe = res["per_probe"]
    bare = [p["bare_labels"] for p in per_probe]
    grounded = [p["grounded_labels"] for p in per_probe]
    cluster = {
        lab: cluster_bootstrap_rate_ci(bare, grounded, lab, n_boot=n_boot, seed=seed)
        for lab in LABELS
    }
    res["cluster_ci"] = {
        "method": (
            f"two-stage cluster bootstrap: resample probes (n={len(per_probe)}) with "
            f"replacement, then decodes within each; B={n_boot}. a=bare, b=grounded."
        ),
        **cluster,
    }
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res["cluster_ci"]


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/main/grounding/results.json")
    cc = recompute(path)
    for lab in LABELS:
        c = cc[lab]
        flag = "delta excludes 0" if c["delta_excludes_zero"] else "delta spans 0 (directional)"
        print(
            f"{lab:12s} bare={c['a']['rate']:.3f} [{c['a']['ci'][0]:.3f},{c['a']['ci'][1]:.3f}]  "
            f"grounded={c['b']['rate']:.3f} [{c['b']['ci'][0]:.3f},{c['b']['ci'][1]:.3f}]  "
            f"Δ={c['delta_b_minus_a']:+.3f} [{c['delta_ci'][0]:+.3f},{c['delta_ci'][1]:+.3f}]  "
            f"disjoint={c['intervals_disjoint']}  {flag}"
        )
    print(f"\nwrote cluster_ci -> {path}")


if __name__ == "__main__":
    main()
