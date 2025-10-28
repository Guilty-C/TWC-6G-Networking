"""PD9 Preflight CLI shim

Bridges the top-level module path `semntn.src.pd9_preflight` to the
implementation in `semntn.src.tools.pd9_preflight` so that the runbook
command `python -m semntn.src.pd9_preflight --config ...` works.
"""
from __future__ import annotations

try:  # Prefer relative import when executed as a module
    from .tools.pd9_preflight import main
except Exception:  # Fallback for script execution contexts
    from semntn.src.tools.pd9_preflight import main


if __name__ == "__main__":
    main()