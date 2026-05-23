"""Smoke tests: verify imports and data presence."""
import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_eval_logger_imports():
    from eval.logger import ExperimentLogger  # noqa: F401


def test_precomputed_r8_pickle_loads():
    p = ROOT / "data" / "precomputed_graphs" / "r8_subword_dep.pkl"
    if not p.exists():
        return
    with open(p, "rb") as f:
        data = pickle.load(f)
    assert data is not None


def test_results_summary_parses():
    import json
    p = ROOT / "results" / "paper_b_R8_5seeds.json"
    with open(p) as f:
        data = json.load(f)
    assert data is not None
