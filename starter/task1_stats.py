import sentencepiece as spm
import matplotlib.pyplot as plt
from pathlib import Path
from tokenizer import read_pairs, BOS_ID, EOS_ID
from embeddings import PositionalEncoding


sp=spm.SentencePieceProcessor(model_file="sql_sp.model")   #loading the trained vocab
maxsrc, maxtgt = 160 , 64   #max lenghts for source and target


def lengths(Path):
    src_len, tgt_len, over_long = [],[],0
    for pair in read_pairs(Path):
        src= sp.encode(pair["src"]) + [EOS_ID]
        tgt= [BOS_ID] + sp.encode(pair["tgt"]) + [EOS_ID]
        src_len.append(len(src))
        tgt_len.append(len(tgt))
        if len(src) > maxsrc or len(tgt) > maxtgt:
            over_long +=1

    return src_len,tgt_len,over_long

for split in ("train","dev","test"):
    src_len, tgt_len, over_long= lengths(f"{split}_pairs.jsonl")
    n =len(src_len)
    print(
        f"{split}: pairs{n}"
        f"source mean={sum(src_len)/n:.2f} max={max(src_len)}  "
        f"target mean={sum(tgt_len)/n:.2f} max={max(tgt_len)}  "
        f"over_long={over_long}"
    )

pe=PositionalEncoding(d_model=256).pe[0, :100, :]   #heatmap includes first 100 positions. defaults: 512 positions, 256 numbers each
out= Path("../results/figures")
out.mkdir(parents=True, exist_ok=True)


fig, ax = plt.subplots(figsize=(10, 4))
image = ax.imshow(pe, aspect="auto", cmap="coolwarm", interpolation="nearest")
ax.set_xlabel("dimension")
ax.set_ylabel("position")
ax.set_title("Positional encoding, first 100 positions")
fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(out / "positional_encoding.png", dpi=150)
print("saved", out / "positional_encoding.png")

