"""Persist every number a notebook produces to JSON.

A figure shows a result; the JSON file *is* the result. Writing both means the article, the
README and any later check all quote the same numbers, and none of them have to be
transcribed by hand from a notebook cell. Each notebook writes one file named after itself
into ``outputs/results/``.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.config import RESULTS_DIR


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands the numpy types estimates come back as."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def sanitise(value: Any) -> Any:
    """Replace non-finite floats with ``None``, recursively.

    This cannot be done in :class:`NumpyJSONEncoder`. ``json`` only calls ``default()`` for
    types it does not already know, and ``np.float64`` subclasses ``float`` — so a NaN
    estimate goes straight down the float path and is written as a bare ``NaN`` literal,
    which is not legal JSON. The payload has to be walked before encoding.
    """
    if isinstance(value, dict):
        return {key: sanitise(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [sanitise(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitise(value.tolist())
    if isinstance(value, np.floating | float):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def results_path(name: str) -> Path:
    """Absolute path for a named results file, creating the output directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{name}.json"


def save_results(name: str, payload: dict[str, Any]) -> Path:
    """Write ``payload`` to ``outputs/results/<name>.json`` and return the path."""
    path = results_path(name)
    with path.open("w", encoding="utf-8") as handle:
        # allow_nan=False so that anything sanitise missed raises here rather than writing
        # a file that a strict parser will reject later.
        json.dump(
            sanitise(payload),
            handle,
            indent=2,
            cls=NumpyJSONEncoder,
            sort_keys=False,
            allow_nan=False,
        )
        handle.write("\n")
    return path


def load_results(name: str) -> dict[str, Any]:
    """Read back a results file. Used by notebook 06 to assemble the final summary."""
    path = results_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no results at {path}; run the notebook that writes it first")
    with path.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded
