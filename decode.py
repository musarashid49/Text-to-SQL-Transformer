
import json
import sys
from pathlib import Path
import torch
STARTER = Path(__file__).resolve().parent / "starter"
if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))
from data_prep import AGG_OPS, COND_OPS, encode_target
from tokenizer import BOS_ID, EOS_ID, PAD_ID
# "max" -> 1, "min" -> 2, ...  Index 0 is "no aggregate", so it has no word.
AGG_INDEX = {word.lower(): i for i, word in enumerate(AGG_OPS) if word}



def column_index(t):
    if len(t) > 5 or not t.startswith("<c") or not t.endswith(">"):
        return None
    num = t[2: -1]    #index 2 till last element but not including that
    if not num.isdigit():
        return None
    return int(num)

def parse_sql(text):
    words= text.strip().lower().split()  #converting to lowercase
    if not words or words[0]!= "select":
        return None
    i = 1
    agg = 0
    if i < len(words) and words[i] in AGG_INDEX:
        agg = AGG_INDEX[words[i]]
        i+=1

    if i >= len(words):
        return None

    sel=column_index(words[i])
    if sel is None:
        return None
    i +=1

    conds=[]
    while i < len(words):
        w= words[i]
        expect= "where" if len(conds)==0 else "and"
        if w != expect :
            return None
        i +=1
        if i >= len(words):
            return None
        col = column_index(words[i])
        if col is None:
            return None
        i += 1
        if i >= len(words) or words[i] not in COND_OPS:
            return None
        op = COND_OPS.index(words[i])
        i += 1

        value_words = []
        while i < len(words):

            starts_next = (
                words[i] == "and"
                and i + 2 < len(words)
                and column_index(words[i + 1]) is not None
                and words[i + 2] in COND_OPS
            )
            if starts_next:
                break
            value_words.append(words[i])
            i += 1
        if not value_words:
            return None
        conds.append([col, op, " ".join(value_words)])

    return {"sel": sel, "agg": agg, "conds": conds}
    

def to_readable_sql(query , header):
    name = header[query["sel"]]
    if query["agg"] == 0:
        select= name
    else:
        
        agg_word = AGG_OPS[query["agg"]]
        select = f"{agg_word} {name}"

    sql = f"SELECT {select} FROM table"
    if not query["conds"]:
        return sql
    parts = []
    for col, op, val in query["conds"]:
        parts.append(f"{header[col]} {COND_OPS[op]} '{val}'")
    return sql + " WHERE " + " AND ".join(parts)


@torch.no_grad()
def greedy_decode(model, source, max_len=64):

    if source.dim() == 1:
        source = source.unsqueeze(0)
    device = next(model.parameters()).device
    source = source.to(device)
    was_training = model.training
    model.eval()
    encoder_output, source_mask = model.encode(source)
    tokens = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        logits, _ = model.decode(tokens, encoder_output, source_mask)
        next_scores = logits[0, -1].clone()
        next_scores[PAD_ID] = float("-inf")
        next_id = int(next_scores.argmax())
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        if next_id == EOS_ID:
            break
    model.train(was_training)
    return tokens.squeeze(0).tolist()


@torch.no_grad()
def beam_decode(model, source, beam_size=4, max_len=64):
    """Keep beam_size unfinished sentences. Return the best id list.

    Scores are sums of log-probabilities. Closer to zero is better.
    Stops at </s>, or after max_len tokens.
    """
    if source.dim() == 1:
        source = source.unsqueeze(0)
    device = next(model.parameters()).device
    source = source.to(device)

    was_training = model.training
    model.eval()
    encoder_output, source_mask = model.encode(source)

    active = [(0.0, [BOS_ID])]
    done = []

    for _ in range(max_len - 1):
        if not active:
            break
        candidates = []
        for score, tokens in active:
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, _ = model.decode(token_tensor, encoder_output, source_mask)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1).clone()
            log_probs[PAD_ID] = float("-inf")
            top_scores, top_ids = torch.topk(log_probs, beam_size)
            for next_score, next_id in zip(top_scores.tolist(), top_ids.tolist()):
                new_tokens = tokens + [next_id]
                new_score = score + next_score
                if next_id == EOS_ID:
                    done.append((new_score, new_tokens))
                else:
                    candidates.append((new_score, new_tokens))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        active = candidates[:beam_size]
        if done and active and active[0][0] <= max(score for score, _ in done):
            break

    model.train(was_training)
    pool = done if done else active
    pool.sort(key=lambda pair: pair[0], reverse=True)
    return pool[0][1]


if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        failed = 0
        n = 0
        shown = 0
        with (STARTER / f"{split}_pairs.jsonl").open(encoding="utf-8") as f:
            for line in f:
                tgt = json.loads(line)["tgt"]
                n += 1
                got = parse_sql(tgt)
                if got is None or encode_target(got) != tgt:
                    failed += 1
                    if shown < 3:
                        print("FAIL", tgt, "->", got)
                        shown += 1
        print(f"{split} round-trip failures: {failed} / {n}")
