# Text-to-SQL with the Transformer

An encoder–decoder Transformer (Vaswani et al., 2017), built and trained from scratch in PyTorch,
that turns an English question about a table into a WikiSQL query.

> Generative AI — Assignment 02 (Fall 2026)

## Repository layout

| Path | Contents |
|---|---|
| `starter/` | the four given files and `check_starter.py`, unchanged |
| `model/attention.py` | scaled dot-product and multi-head attention |
| `model/layers.py` | feed-forward, encoder layer, decoder layer |
| `model/transformer.py` | full model and masks |
| `train.py` | Task 3 – training |
| `decode.py` | greedy, beam search, parser, readable SQL |
| `evaluate_model.py` | prediction files, component accuracy, attention map |
| `results/` | prediction files, tables, figures, `samples.md` |
| `app/` | front end |

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/salesforce/WikiSQL
cd WikiSQL && tar xvjf data.tar.bz2 && cd ..
cd starter && ln -s ../WikiSQL WikiSQL
```

## Task 1 – Run the starter code

From inside `starter/`:

```bash
python data_prep.py
python tokenizer.py
python check_starter.py
```

Expected output of `check_starter.py` (batch lengths vary):

```
train_pairs.jsonl: kept 56336, skipped 19
dev_pairs.jsonl: kept 8421, skipped 0
src (64, 123) tgt (64, 31)
encoder input (64, 123, 256) decoder input (64, 30, 256)
```

## Model configuration

| | |
|---|---|
| d_model | 256 |
| Heads h | 4 (d_k = d_v = 64) |
| Encoder / decoder layers | 3 / 3 |
| d_ff | 1024 |
| Dropout | 0.1 |
| Normalisation | post-norm, LayerNorm(x + Sublayer(x)) |
| Weight sharing | encoder embedding = decoder embedding = output projection |

## Training

Run the required single 20-epoch training run on a Colab or Kaggle GPU:

```bash
python train.py
```

Epoch losses are weighted by non-padding target-token counts; the per-batch
training objective is unchanged. The lowest dev loss selects `results/best.pt`.
`results/last.pt` stores the latest completed epoch for recovery:

```bash
python train.py --resume results/last.pt
```

Resume continues through epoch 20, rather than starting another 20 epochs. Keep
the run's `best.pt` alongside `last.pt`, so the previously selected checkpoint
remains available. Checkpoints include optimizer and random-generator states,
seed (42), configuration, tokenizer SHA-256, epoch history, device and PyTorch
version. Use the same data, tokenizer, hardware and environment when resuming;
identical results across devices or nondeterministic kernels are not guaranteed.
Older checkpoints without this metadata remain usable for evaluation but cannot
be resumed by this script.

Each completed epoch writes `results/training_history.json` (losses, step, LR,
and elapsed seconds), `results/figures/training_loss.png`, and
`results/figures/learning_rate.png` (the prescribed first 20,000 steps).

## Evaluation

From inside `WikiSQL/`:

```bash
python evaluate.py data/dev.jsonl data/dev.db ../results/dev_greedy.jsonl
```

## Results

### Table 1 – Data

| | Train | Dev | Test |
|---|---|---|---|
| Pairs | | | |
| Mean / max source length (tokens) | | | |
| Mean / max target length (tokens) | | | |
| Pairs dropped as too long | | – | – |

### Table 2 – Model and training

| | |
|---|---|
| Trainable parameters | 7,577,600 |
| Epochs trained / best epoch | |
| Best dev loss | |
| Training time and GPU | |

### Table 3 – Official metrics

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | | | |
| Dev | beam (4) | | | |
| Test | | | | |

### Table 4 – Component accuracy (dev)

| | |
|---|---|
| `sel` column correct (%) | |
| `agg` correct (%) | |
| `WHERE` clause correct (%) | |

## Correctness checks

- [ ] Causal mask
- [ ] Padding mask
- [ ] Attention rows sum to 1
- [ ] Weight sharing (`is` check)
- [ ] Learning-rate schedule plot
- [ ] Gold round-trip (execution accuracy > 99%)

## Front end

_TODO: screenshot_

## Links

- Medium blog: _TODO_
- LinkedIn post: _TODO_
