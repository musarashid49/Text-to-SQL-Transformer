"""Importing the model package puts starter/ on sys.path, so the given starter
files (tokenizer, embeddings, dataset, ...) can be imported unchanged from
anywhere in the repo."""

import sys
from pathlib import Path

_STARTER = Path(__file__).resolve().parent.parent / "starter"
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))
