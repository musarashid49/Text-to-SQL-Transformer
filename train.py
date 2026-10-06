import sys
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
from dataset import make_loader
from tokenizer import PAD_ID
from model.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print("device", device)

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "starter" / "sql_sp.model"))
train_loader = make_loader(str(ROOT / "starter" / "train_pairs.jsonl"), sp, train=True, batch_size=64)  #dropping longer pairs
dev_loader = make_loader(str(ROOT / "starter" / "dev_pairs.jsonl"), sp, train=False, batch_size=64)   #no dropping longer pairs
model = Transformer(sp.get_piece_size()).to(device)
print("trainable parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing= 0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0 , betas=(0.9, 0.98) , eps=1e-9)

def learning_rate(step, d_model=256, warmup=4000):
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

def run_epoch(model, loader, optimizer, step, training):
    model.train(training)
    total_loss = 0.0
    n_batches = 0
    lr_now = optimizer.param_groups[0]["lr"]

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if training:
            loss.backward()
            step += 1
            lr_now = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches, step, lr_now

def main():
    step = 0
    best_dev = float("inf")
    save_path = ROOT / "results" / "best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 21):
        train_loss, step, lr_now = run_epoch(model, train_loader, optimizer, step, training=True)
        dev_loss, step, lr_now = run_epoch(model, dev_loader, optimizer, step, training=False)
        print(f"epoch {epoch}  train {train_loss:.4f}  dev {dev_loss:.4f}  lr {lr_now:.6g}")

        if dev_loss < best_dev:
            best_dev = dev_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "dev_loss": dev_loss,
                "step": step,
            }, save_path)
            print("saved", save_path, "epoch", epoch)


if __name__ == "__main__":
    main()
