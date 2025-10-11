"""Extremely small YAML reader for configuration files."""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if text.startswith("0") and text not in {"0", "0.0"}:
            return ast.literal_eval(text)
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            pass
    if text.startswith("[") or text.startswith("{"):
        return ast.literal_eval(text)
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        return ast.literal_eval(text)
    return text


def load(path: str) -> Dict[str, Any]:
    stack: List[Tuple[Dict[str, Any], int]] = [({}, -1)]
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            while stack and indent <= stack[-1][1]:
                stack.pop()
            if not stack:
                raise ValueError(f"Invalid indentation in line: {raw_line}")
            current = stack[-1][0]
            if value == "":
                new_dict: Dict[str, Any] = {}
                current[key] = new_dict
                stack.append((new_dict, indent))
            else:
                current[key] = _parse_scalar(value)
    return stack[0][0]


__all__ = ["load"]
