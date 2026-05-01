from __future__ import annotations

from json import loads as _json_loads


def safe_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = _json_loads(value)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    return {}
