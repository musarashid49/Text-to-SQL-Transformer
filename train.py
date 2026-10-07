import sys
import argparse
import hashlib
import json
import random
import time
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
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing= 0.1)
MODEL_CONFIG = dict(d_model=256, h=4, n_layers=3, d_ff=1024, dropout=0.1)
TRAIN_CONFIG = dict(batch_size=64, epochs=20, warmup_steps=4000,
                    label_smoothing=0.1, betas=(0.9, 0.98), eps=1e-9)
SEED = 42

def learning_rate(step, d_model=256, warmup=4000):
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

def run_epoch(model, loader, optimizer, step, training):
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
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

        n_tokens = (tgt_out != PAD_ID).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens, step, lr_now


def rng_state():
    state = {"python": random.getstate(), "cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and device.type == "mps":
        torch.mps.set_rng_state(state["mps"])


def write_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    epochs = [row["epoch"] for row in history]
    for name in ("train_loss", "dev_loss"):
        ax.plot(epochs, [row[name] for row in history], label=name)
    ax.set(xlabel="Epoch", ylabel="Loss per non-padding target token")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "training_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    steps = range(1, 20001)
    ax.plot(steps, [learning_rate(step) for step in steps])
    ax.set(xlabel="Optimizer step", ylabel="Learning rate")
    fig.tight_layout()
    fig.savefig(figures / "learning_rate.png", dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, help="Resume an epoch-end checkpoint")
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "mps":
        torch.mps.manual_seed(SEED)
    print("device", device)
    tokenizer_path = ROOT / "starter" / "sql_sp.model"
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    train_loader = make_loader(str(ROOT / "starter" / "train_pairs.jsonl"), sp, train=True, batch_size=64)
    dev_loader = make_loader(str(ROOT / "starter" / "dev_pairs.jsonl"), sp, train=False, batch_size=64)
    model = Transformer(sp.get_piece_size(), **MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    print("trainable parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    step = 0
    start_epoch = 1
    history = []
    best_dev = float("inf")
    save_path = ROOT / "results" / "best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (checkpoint["tokenizer_sha256"] != tokenizer_hash
                or checkpoint["model_config"] != MODEL_CONFIG
                or checkpoint["train_config"] != TRAIN_CONFIG):
            raise ValueError("Checkpoint tokenizer or configuration does not match this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        start_epoch = checkpoint["epoch"] + 1
        best_dev = checkpoint["best_dev_loss"]
        history = checkpoint["history"]
        restore_rng(checkpoint["rng_state"])
        print("resuming after epoch", checkpoint["epoch"])

    for epoch in range(start_epoch, 21):
        started = time.perf_counter()
        train_loss, step, lr_now = run_epoch(model, train_loader, optimizer, step, training=True)
        dev_loss, step, lr_now = run_epoch(model, dev_loader, optimizer, step, training=False)
        print(f"epoch {epoch}  train {train_loss:.4f}  dev {dev_loss:.4f}  lr {lr_now:.6g}")

        history.append(dict(epoch=epoch, step=step, train_loss=train_loss,
                            dev_loss=dev_loss, learning_rate=lr_now,
                            seconds=time.perf_counter() - started))
        improved = dev_loss < best_dev
        if improved:
            best_dev = dev_loss
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "dev_loss": dev_loss,
            "step": step,
            "best_dev_loss": best_dev,
            "model_config": MODEL_CONFIG,
            "train_config": TRAIN_CONFIG,
            "seed": SEED,
            "tokenizer_sha256": tokenizer_hash,
            "rng_state": rng_state(),
            "history": history,
            "device": str(device),
            "torch_version": str(torch.__version__),
        }
        torch.save(checkpoint, save_path.parent / "last.pt")
        if improved:
            torch.save(checkpoint, save_path)
            print("saved", save_path, "epoch", epoch)
        write_history(history, save_path.parent)


if __name__ == "__main__":
    main()
