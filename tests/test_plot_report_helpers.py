from __future__ import annotations

from scripts.plot_report import collect_deltas, leak_rate, machinery_pairs


def _delta(d: float, lo: float, hi: float, ez: bool) -> dict:
    return {"delta": d, "ci_lo": lo, "ci_hi": hi, "excludes_zero": ez, "favored": "b"}


def test_collect_deltas_orders_and_extracts() -> None:
    runs = {
        "main": {
            "scorecard": {
                "deltas_api_minus_lora": {"len_wasserstein": _delta(125.9, 107.6, 142.4, True)}
            }
        },
        "armA": {
            "scorecard": {
                "deltas_api_minus_lora": {"len_wasserstein": _delta(3.57, 1.53, 4.66, True)}
            }
        },
    }
    rows = collect_deltas(runs, "len_wasserstein", order=["main", "armA"])
    assert [r["name"] for r in rows] == ["main", "armA"]
    assert rows[0]["delta"] == 125.9 and rows[0]["excludes_zero"] is True


def test_leak_rate_from_guard() -> None:
    on = {"retrieval_leak_guard": {"id_leaks": 17}, "scorecard": {"n_items": 60}}
    off = {"retrieval_leak_guard": {"id_leaks": 0}, "scorecard": {"n_items": 60}}
    assert leak_rate(on) == (17, 60)
    assert leak_rate(off) == (0, 60)


def test_machinery_pairs_api_side() -> None:
    main = {
        "scorecard": {
            "arms": {
                "api": {"len_wasserstein_vs_real": 128.8, "exclaim_rate": 0.651},
                "lora": {"len_wasserstein_vs_real": 2.9, "exclaim_rate": 0.0},
            }
        }
    }
    arma = {
        "scorecard": {
            "arms": {
                "api": {"len_wasserstein_vs_real": 6.97, "exclaim_rate": 0.0},
                "lora": {"len_wasserstein_vs_real": 3.41, "exclaim_rate": 0.0},
            }
        }
    }
    out = machinery_pairs(main, arma, "len_wasserstein_vs_real")
    assert out["api"] == (128.8, 6.97) and out["lora"] == (2.9, 3.41)
