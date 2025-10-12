import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from semntn.src.regret.runner import run

def test_regret_smoke(tmp_path):
    cfg = {
        "env": {"type": "bernoulli", "K": 3, "T": 2000, "means": [0.7, 0.6, 0.55]},
        "seeds": [1, 2],
        "baselines": ["ucb", "eps", "rand", "oracle"],
        "ucb": {"alpha": 1.5},
        "epsilon": 0.1,
        "plot": {"logx": False, "show_ref_sqrt": True},
    }
    res = run(cfg, outdir=str(tmp_path), make_plots=False, save_csv=False)
    assert "slope" in res and "C*" in res and "pass" in res
    # monotonic non-decreasing R(t) guaranteed by construction; ensure not NaN:
    assert res["slope"] == res["slope"]
    assert res["C*"] == res["C*"]
