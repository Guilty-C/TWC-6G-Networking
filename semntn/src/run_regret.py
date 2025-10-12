import argparse
import json
import os
import sys

import yaml

# ensure local imports when run as script
sys.path.append(os.path.dirname(__file__))

from regret.runner import run_with_timestamp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/regret.yaml")
    ap.add_argument("--outdir", default=None, help="Base output directory")
    ap.add_argument("--no-plots", action="store_true", help="Disable plotting")
    ap.add_argument("--save-csv", action="store_true", help="Force saving CSV curves")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_outdir = args.outdir or cfg.get("outdir", "outputs/regret")
    make_plots = not args.no_plots
    save_csv = args.save_csv or bool(cfg.get("save_csv", True))

    os.makedirs(base_outdir, exist_ok=True)

    result = run_with_timestamp(cfg, base_outdir, make_plots=make_plots, save_csv=save_csv)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
