"""Task 5: evaluation and analysis.

Spec:
  - Write prediction files to results/ (dev greedy, dev beam, test final).
    Then run the OFFICIAL evaluator from inside WikiSQL/:
        python evaluate.py data/dev.jsonl data/dev.db ../results/dev_greedy.jsonl
  - Component accuracy on dev: % correct sel column, % correct agg,
    % correct WHERE clause (same set of conditions, order ignored).
  - Parse-failure rate: % of lines written as {"error": "parse"}.
  - Attention map: for one dev example, plot last-layer decoder cross-attention
    (averaged over heads), generated tokens x source tokens.
  - Correctness checks (Section 3.2) - include output in README.
"""

# TODO: implement
