import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from semntn.src.regret.runner import run

def test_regret_smoke(tmp_path):
    cfg = {
        "seed": 7,
        "out_dir": str(tmp_path / "out"),
        "env": {"type": "bernoulli", "K": 3, "T": 5000, "means": [0.7, 0.6, 0.55]},
        "algo": "ucb1",
        "ucb1": {"alpha": 2.0},
        "eg": {"epsilon": 0.1},
        "sem_ucb": {"alpha": 2.0, "lambda0": 0.3, "decay": "log", "c_decay": 1.0, "predictor": "none", "lr": 0.05},
        "plot": {"dpi": 80, "show": False},
        "accept": {"slope_logT_range": [0.1, 2000.0], "cstar_max": 100.0},
    }
    res = run(cfg)
    assert "slope" in res and "C*" in res and "pass" in res
    # monotonic non-decreasing R(t) guaranteed by construction; ensure not NaN:
    assert res["slope"] == res["slope"]
    assert res["C*"] == res["C*"]
