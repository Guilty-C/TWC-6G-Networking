#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv311
source .venv311/bin/activate

python -m pip install --upgrade pip
python -m pip install -r semntn/requirements.txt

python semntn/src/run_sem_eval.py --config semntn/configs/sem_eval.yaml
