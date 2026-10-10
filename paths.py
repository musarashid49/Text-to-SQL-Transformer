"""Shared paths. Importing this also puts starter/ on sys.path, so the given
files can be imported unchanged (from tokenizer import PAD_ID, ...) no matter
which directory a script is run from."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "starter"
WIKISQL = ROOT / "WikiSQL"
WIKISQL_DATA = WIKISQL / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CHECKPOINTS = ROOT / "checkpoints"

SP_MODEL = STARTER / "sql_sp.model"


def pairs_path(split):
    """starter/{train,dev,test}_pairs.jsonl, written by starter/data_prep.py."""
    return STARTER / f"{split}_pairs.jsonl"


if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))
