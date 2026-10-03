"""Task 4: decoding and conversion to SQL.

Spec:
  - Greedy decoding and beam search (beam size 4); stop at </s> or 64 tokens.
  - Detokenise with sp.decode and parse into WikiSQL format:
        {"sel": int, "agg": int, "conds": [[col, op, value], ...]}
    using AGG_OPS and COND_OPS from starter/data_prep.py.
  - Write one JSON line per dev/test example, in the same order as dev.jsonl:
        {"query": {...}}   or   {"error": "parse"} if unparseable. Never skip a line.
  - Function that turns a parsed query into readable SQL with real column names.
"""

# TODO: implement
