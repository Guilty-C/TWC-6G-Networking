from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "semntn" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import inner_api_ucb
from wer_proxy import per_from_snr_db


class TestWerProxy(unittest.TestCase):
    def test_per_monotonic(self) -> None:
        snr_values = [-5, -2, 0, 2, 5]
        per_values = [per_from_snr_db(x) for x in snr_values]
        for a, b in zip(per_values, per_values[1:]):
            self.assertGreaterEqual(a, b)

    def test_per_bounds(self) -> None:
        self.assertLess(per_from_snr_db(50), 1e-3)
        self.assertGreater(per_from_snr_db(-50), 0.99)


class TestUCBInner(unittest.TestCase):
    def setUp(self) -> None:
        inner_api_ucb.reset_state()

    def _step(self, pick: dict, ctx: dict, action_space: dict) -> None:
        obs_ctx = dict(ctx)
        obs_ctx.update(
            {
                "S_bits_obs": pick["S_bits"],
                "E_obs": pick["E_hat"],
                "swwer_obs": pick["SWWER_hat"],
            }
        )
        inner_api_ucb.pick_action_and_estimate(obs_ctx, action_space, 0.0, 0.0, 10, 1.0)

    def test_initial_exploration(self) -> None:
        action_space = {"tuples": [(300, 0.2, 1), (300, 0.4, 2)]}
        ctx = {"snr_db": 0.0, "slot_sec": 0.02}
        pick1 = inner_api_ucb.pick_action_and_estimate(ctx, action_space, 0.0, 0.0, 10, 1.0)
        self._step(pick1, ctx, action_space)
        pick2 = inner_api_ucb.pick_action_and_estimate(ctx, action_space, 0.0, 0.0, 10, 1.0)
        self.assertNotEqual(pick1["action"][:3], pick2["action"][:3])

    def test_rewards_stay_in_bounds(self) -> None:
        action_space = {"q_bps": [300], "p_grid_w": [0.2], "b_subbands": [1]}
        ctx = {"snr_db": 0.0, "slot_sec": 0.02}
        for _ in range(5):
            pick = inner_api_ucb.pick_action_and_estimate(ctx, action_space, 0.0, 0.0, 10, 1.0)
            self._step(pick, ctx, action_space)
        arm = inner_api_ucb.pick_action_and_estimate(ctx, action_space, 0.0, 0.0, 10, 1.0)
        self.assertLessEqual(arm["SWWER_hat"], 1.0)
        self.assertGreaterEqual(arm["SWWER_hat"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
