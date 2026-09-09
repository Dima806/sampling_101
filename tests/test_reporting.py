"""Results persistence: what the notebooks write must be valid, strict JSON.

The JSON files are the project's record of its own numbers. If they cannot be parsed by a
strict reader, the article and the README are quoting something nothing can verify.
"""

import json

import numpy as np
import pytest

from src.reporting import NumpyJSONEncoder, load_results, results_path, save_results


def test_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.reporting.RESULTS_DIR", tmp_path)
    payload = {"notebook": "test", "value": 1.5, "counts": [1, 2, 3]}
    save_results("scratch", payload)
    assert load_results("scratch") == payload


def test_numpy_scalars_and_arrays_survive(tmp_path, monkeypatch) -> None:
    """Estimates come back as numpy scalars, which plain json.dump refuses to encode."""
    monkeypatch.setattr("src.reporting.RESULTS_DIR", tmp_path)
    payload = {
        "float": np.float64(0.25),
        "int": np.int64(7),
        "bool": np.bool_(True),
        "array": np.arange(3),
    }
    save_results("numpy", payload)
    loaded = load_results("numpy")
    assert loaded == {"float": 0.25, "int": 7, "bool": True, "array": [0, 1, 2]}


def test_non_finite_values_become_null(tmp_path, monkeypatch) -> None:
    """NaN and Infinity are not legal JSON.

    ``max_weight_share`` is NaN whenever a proposal never reaches the event, which is a
    real outcome in notebook 05. Emitting a bare NaN would produce a file that strict
    parsers reject, so it is recorded as null instead.
    """
    monkeypatch.setattr("src.reporting.RESULTS_DIR", tmp_path)
    save_results("nonfinite", {"nan": np.float64("nan"), "inf": np.float64("inf")})

    raw = results_path("nonfinite").read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    # json.loads accepts NaN by default; parse strictly to prove a real parser would too.
    assert json.loads(raw, parse_constant=_reject) == {"nan": None, "inf": None}


def _reject(value: str) -> float:
    raise AssertionError(f"non-finite literal {value!r} reached the file")


def test_unsupported_type_still_raises(tmp_path, monkeypatch) -> None:
    """The encoder widens what is accepted; it must not silently swallow real mistakes."""
    monkeypatch.setattr("src.reporting.RESULTS_DIR", tmp_path)
    with pytest.raises(TypeError):
        save_results("bad", {"set": {1, 2, 3}})


def test_missing_results_file_explains_itself() -> None:
    with pytest.raises(FileNotFoundError, match="run the notebook"):
        load_results("definitely_not_a_real_notebook")


def test_encoder_handles_paths() -> None:
    assert json.dumps({"p": results_path("x")}, cls=NumpyJSONEncoder).endswith('x.json"}')
