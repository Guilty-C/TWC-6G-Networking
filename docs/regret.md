# Regret Evaluation Pipeline

This repository provides a reproducible pipeline for drawing cumulative pseudo-regret
curves for the simple bandit environments under `semntn/src/regret/`.

## Regret definition & normalisation

* **Pseudo-regret** is defined as
  \[R_T = \sum_{t=1}^{T} (\max_a \mu_{\text{true}}(a) - \mu_{\text{true}}(a_t)).\]
* The expectation \(\mu_{\text{true}}(a)\) is computed using the exact same linear
  mapping to the unit interval as the learners.  The helper
  `map_to_unit_interval` in `semntn/src/regret/envs.py` is the single source of
  truth for this scaling.

## Baselines

The runner evaluates multiple strategies over several random seeds and aggregates
mean and 95% confidence intervals:

* `ucb` – classic UCB1 policy operating on the normalised rewards.
* `eps` – ε-greedy with configurable exploration rate.
* `rand` – uniform random policy.
* `oracle` – plays the action with the highest \(\mu_{\text{true}}\).

## Running experiments

```bash
python semntn/src/run_regret.py --config configs/regret.yaml
```

Key CLI flags:

* `--outdir` – base directory for results (default from config).
* `--no-plots` – skip generating plots.
* `--save-csv` – force saving the aggregated CSV curve.

Outputs are stored in `outputs/regret/<timestamp>/` by default and contain the
plot (`regret.png`), aggregated statistics (`regret.csv`), and a `summary.json`
file with diagnostics.

## Acceptance & self-check

The script prints a JSON line with
`{"slope": <float>, "C*": <float>, "pass": true/false}`.  Passing requires:

* Pearson correlation between \(R(T)\) and \(\sqrt{T}\) ≥ 0.98.
* UCB mean regret lower than Random/ε-Greedy and higher than the Oracle.

These checks prevent the staircase/plateau artefacts observed previously and
ensure the curves are sub-linear and concave as expected.
