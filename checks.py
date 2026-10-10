"""Section 3.2 correctness checks. Run from the repo root:  python checks.py
Paste the output into the README. More checks are added as each phase lands."""

import math
import sys
import torch
from model.attention import ScaledDotProductAttention, MultiHeadAttention
from model.layers import EncoderLayer, DecoderLayer, PositionwiseFeedForward
from model.transformer import Encoder, Decoder, Transformer
from tokenizer import PAD_ID                     # starter/ is on sys.path via model/__init__.py

VOCAB = 8000

D_MODEL, HEADS, D_K, D_FF = 256, 4, 64, 1024


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def check_attention_rows_sum_to_one():
    """Every attention row sums to 1 over the unmasked keys, and puts ~0 weight
    on masked ones. Two sequences of length 7 and 4 padded to 7 (self-attention,
    so query_position and key_position have the same size)."""
    torch.manual_seed(0)
    lengths = torch.tensor([7, 4])
    batch, positions = len(lengths), int(lengths.max())
    q, k, v = (torch.randn(batch, HEADS, positions, D_K) for _ in range(3))
    mask = (torch.arange(positions)[None, :] < lengths[:, None])[:, None, None, :]   # (B, 1, 1, key_position)

    out, w = ScaledDotProductAttention()(q, k, v, mask)
    assert out.shape == (batch, HEADS, positions, D_K), f"output shape {tuple(out.shape)}"
    assert w.shape == (batch, HEADS, positions, positions), f"weights shape {tuple(w.shape)}"

    row_err = (w.sum(-1) - 1).abs().max().item()
    masked_w = w.masked_select(~mask.expand_as(w)).abs().max().item()
    return report("attention rows sum to 1", row_err < 1e-5 and masked_w < 1e-6,
                  f"max |row sum - 1| = {row_err:.1e}, max weight on masked keys = {masked_w:.1e}")


def n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_multi_head_attention():
    """Cross-attention shapes (3 queries over 8 keys), parameter count, and
    every projection matrix actually initialised (torch.empty leaves garbage)."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, HEADS).eval()
    out, w = mha(torch.randn(2, 3, D_MODEL), torch.randn(2, 8, D_MODEL), torch.randn(2, 8, D_MODEL))
    shapes_ok = out.shape == (2, 3, D_MODEL) and w.shape == (2, HEADS, 3, 8)

    bound = math.sqrt(6 / (2 * D_MODEL))
    bad_init = [name for name in ("w_q", "w_k", "w_v", "w_o")
                if not (getattr(mha, name).abs().max() <= bound
                        and getattr(mha, name).std() > 0.5 * bound / math.sqrt(3))]
    count = n_params(mha)
    return report("multi-head attention", shapes_ok and not bad_init and count == 263_168,
                  f"out {tuple(out.shape)}, weights {tuple(w.shape)}, params {count:,}, "
                  f"uninitialised: {bad_init or 'none'}")


def check_layers():
    """FFN / encoder / decoder shapes and parameter counts; the decoder must
    react to the encoder output (cross-attention wired in) and must not see
    future target tokens (given a causal mask)."""
    torch.manual_seed(0)
    ffn, enc, dec = PositionwiseFeedForward(D_MODEL, D_FF), EncoderLayer().eval(), DecoderLayer().eval()
    source, target = torch.randn(2, 9, D_MODEL), torch.randn(2, 5, D_MODEL)
    source_mask = torch.ones(2, 1, 1, 9, dtype=torch.bool)
    source_mask[1, ..., 6:] = False                                       # second sentence: 3 pads
    target_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))[None, None]

    encoded = enc(source, source_mask)
    decoded, cross = dec(target, encoded, target_mask, source_mask)
    shapes_ok = (ffn(target).shape == target.shape and encoded.shape == source.shape
                 and decoded.shape == target.shape and cross.shape == (2, HEADS, 5, 9))
    counts = (n_params(ffn), n_params(enc), n_params(dec))

    uses_encoder = not torch.allclose(dec(target, encoded + 1, target_mask, source_mask)[0], decoded)
    changed = target.clone()
    changed[:, -1] = torch.randn(2, D_MODEL)
    no_peeking = torch.allclose(dec(changed, encoded, target_mask, source_mask)[0][:, :-1],
                                decoded[:, :-1], atol=1e-5)
    ok = shapes_ok and counts == (525_568, 789_760, 1_053_440) and uses_encoder and no_peeking
    return report("encoder/decoder layers", ok,
                  f"params ffn/enc/dec = {counts[0]:,}/{counts[1]:,}/{counts[2]:,}, "
                  f"decoder uses encoder output: {uses_encoder}, no peeking at future: {no_peeking}")


def check_stacks():
    """Encoder and decoder stacks: parameter counts, and EVERY layer actually
    runs (a misplaced `return` inside the loop silently skips layers 2 and 3)."""
    torch.manual_seed(0)
    enc, dec = Encoder().eval(), Decoder().eval()
    ran = []
    for name, stack in (("enc", enc), ("dec", dec)):
        for i, layer in enumerate(stack.layers):
            layer.register_forward_hook(lambda m, args, out, tag=f"{name}{i + 1}": ran.append(tag))

    source_mask = torch.ones(2, 1, 1, 9, dtype=torch.bool)
    target_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))[None, None]
    encoded = enc(torch.randn(2, 9, D_MODEL), source_mask)
    decoded, cross = dec(torch.randn(2, 5, D_MODEL), encoded, target_mask, source_mask)

    expected = [f"enc{i}" for i in (1, 2, 3)] + [f"dec{i}" for i in (1, 2, 3)]
    counts = (n_params(enc), n_params(dec))
    ok = ran == expected and counts == (2_369_280, 3_160_320) and cross.shape == (2, HEADS, 5, 9)
    return report("encoder/decoder stacks", ok,
                  f"params enc/dec = {counts[0]:,}/{counts[1]:,}, layers that ran: {' '.join(ran)}")


def full_model():
    torch.manual_seed(0)
    return Transformer(VOCAB).eval()             # eval(): dropout off, so outputs are repeatable


def random_ids(batch, length):
    return torch.randint(4, VOCAB, (batch, length))   # ids 0-3 are <pad>, <unk>, <s>, </s>


def check_causal_mask():
    """Section 3.2: change the LAST token of the decoder input; the outputs at
    all earlier positions must not change."""
    model = full_model()
    source, target_in = random_ids(2, 12), random_ids(2, 8)
    before = model(source, target_in)
    changed = target_in.clone()
    changed[:, -1] = (changed[:, -1] + 1) % VOCAB
    after = model(source, changed)
    diff_earlier = (after[:, :-1] - before[:, :-1]).abs().max().item()
    diff_last = (after[:, -1] - before[:, -1]).abs().max().item()
    return report("causal mask", diff_earlier < 1e-5 and diff_last > 0,
                  f"max change at earlier positions = {diff_earlier:.1e} (last position changed by {diff_last:.2f})")


def check_padding_mask():
    """Section 3.2: add extra <pad> tokens to a source; the output must not change."""
    model = full_model()
    source, target_in = random_ids(2, 12), random_ids(2, 8)
    padded = torch.cat([source, torch.full((2, 7), PAD_ID)], dim=1)
    diff = (model(padded, target_in) - model(source, target_in)).abs().max().item()
    return report("padding mask", diff < 1e-4,
                  f"source length 12 -> 19 with 7 extra <pad>: max output change = {diff:.1e}")


def check_weight_sharing():
    """Section 3.2: output projection weight and embedding weight are the same tensor."""
    model = full_model()
    shared = model.shared_embedding.emb.weight
    same = (model.output_projection.weight is shared
            and model.encoder_input.tok.emb.weight is shared
            and model.decoder_input.tok.emb.weight is shared)
    return report("weight sharing", same,
                  f"output_projection.weight is embedding weight: {model.output_projection.weight is shared}, "
                  f"encoder & decoder inputs use it too: {same}")


def check_parameter_count():
    """Table 2. Shared embedding 2,048,000 + encoder 2,369,280 + decoder 3,160,320."""
    count = n_params(full_model())
    return report("trainable parameters", count == 7_577_600, f"{count:,}")


if __name__ == "__main__":
    results = [check_attention_rows_sum_to_one(),
               check_multi_head_attention(),
               check_layers(),
               check_stacks(),
               check_causal_mask(),
               check_padding_mask(),
               check_weight_sharing(),
               check_parameter_count()]
    sys.exit(0 if all(results) else 1)
