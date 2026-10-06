
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))

from data_prep import encode_source, load_split
from dataset import make_loader
from tokenizer import EOS_ID, PAD_ID
from model.transformer import Transformer
from decode import beam_decode, greedy_decode, parse_sql

# Set True only after dev scores are in, and only for the decoding method you keep.
RUN_TEST = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp = spm.SentencePieceProcessor(model_file=str(ROOT / "starter" / "sql_sp.model"))
model = Transformer(sp.get_piece_size()).to(device)

ckpt = torch.load(ROOT / "results" / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print("loaded epoch", ckpt["epoch"], "dev loss", ckpt["dev_loss"])


def same_where(pred_conds, gold_conds):
    def pack(conds):
        return set((col, op, str(val).lower()) for col, op, val in conds)
    return pack(pred_conds) == pack(gold_conds)


def write_predictions(split, decode_fn, out_path):
    examples, tables = load_split(split)
    loader = make_loader(
        str(ROOT / "starter" / f"{split}_pairs.jsonl"),
        sp, train=False, batch_size=64,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    failures = 0
    sel_ok = 0
    agg_ok = 0
    where_ok = 0

    with out_path.open("w", encoding="utf-8") as f:
        for src, tgt in loader:
            for row in src:
                row = row[row != PAD_ID]
                ids = decode_fn(model, row)
                query = parse_sql(sp.decode(ids))
                gold = examples[n]["sql"]

                if query is None:
                    failures += 1
                    f.write(json.dumps({"error": "parse"}) + "\n")
                else:
                    f.write(json.dumps({"query": query}) + "\n")
                    if query["sel"] == gold["sel"]:
                        sel_ok += 1
                    if query["agg"] == gold["agg"]:
                        agg_ok += 1
                    if same_where(query["conds"], gold["conds"]):
                        where_ok += 1
                n += 1

    print(split, out_path.name)
    print("examples", n, "of", len(examples))
    print("parse failures %", 100 * failures / n)
    print("sel %", 100 * sel_ok / n)
    print("agg %", 100 * agg_ok / n)
    print("where %", 100 * where_ok / n)


def plot_attention(example_index=0):
    examples, tables = load_split("dev")
    header = tables[examples[example_index]["table_id"]]["header"]
    question = examples[example_index]["question"]
    text = encode_source(question, header)
    src_ids = sp.encode(text) + [EOS_ID]
    src = torch.tensor(src_ids, dtype=torch.long)

    gen_ids = greedy_decode(model, src)
    target = torch.tensor([gen_ids[:-1]], dtype=torch.long, device=device)
    source = src.unsqueeze(0).to(device)

    encoder_output, source_mask = model.encode(source)
    _, cross_weights = model.decode(target, encoder_output, source_mask)
    attn = cross_weights[0].mean(dim=0).cpu()

    src_labels = [sp.id_to_piece(i) for i in src_ids]
    gen_labels = [sp.id_to_piece(i) for i in gen_ids[1:]]

    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_labels)))
    ax.set_xticklabels(src_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(gen_labels)))
    ax.set_yticklabels(gen_labels, fontsize=6)
    ax.set_xlabel("source token")
    ax.set_ylabel("generated token")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "cross_attention.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)
    print(question)
    print(sp.decode(gen_ids))


if __name__ == "__main__":
    write_predictions("dev", greedy_decode, ROOT / "results" / "dev_greedy.jsonl")
    write_predictions("dev", beam_decode, ROOT / "results" / "dev_beam.jsonl")
    plot_attention(0)
    if RUN_TEST:
        write_predictions("test", greedy_decode, ROOT / "results" / "test_greedy.jsonl")
