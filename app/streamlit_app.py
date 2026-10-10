from __future__ import annotations

import copy
import sys
from pathlib import Path

import sentencepiece as spm
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decode import beam_decode, greedy_decode, parse_sql, to_readable_sql
from model.transformer import Transformer
from paths import SP_MODEL
from starter.data_prep import MAX_COLS, encode_source
from starter.tokenizer import EOS_ID

CHECKPOINT = ROOT / "results" / "inference.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model_and_tokenizer():
    missing = [str(path.relative_to(ROOT)) for path in (CHECKPOINT, SP_MODEL) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    device = get_device()
    tokenizer = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    model = Transformer(tokenizer.get_piece_size()).to(device)
    weights = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    return model, tokenizer, device


st.set_page_config(page_title="Text-to-SQL Transformer", page_icon="🧾", layout="centered")
st.title("Text-to-SQL Transformer")
st.write("Turn a question about a table into SQL using the trained WikiSQL Transformer.")

try:
    model, tokenizer, device = load_model_and_tokenizer()
except (FileNotFoundError, KeyError, RuntimeError, OSError) as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info("Place results/inference.pt and starter/sql_sp.model in the project before starting the app.")
    st.stop()

st.caption(f"Model ready · {device.type.upper()}")

with st.form("text_to_sql"):
    question = st.text_input(
        "Question",
        placeholder="e.g. How many players are from Canada?",
    )
    columns_text = st.text_input(
        "Table columns (comma-separated)",
        placeholder="e.g. Player, Country, Goals",
        help="Enter the table's full column list in its original order; column positions must match the table.",
    )
    decoding = st.selectbox("Decoding", ("Beam search (4)", "Greedy"))
    submitted = st.form_submit_button("Generate SQL", type="primary")

if submitted:
    header = [column.strip() for column in columns_text.split(",") if column.strip()]
    if not question.strip():
        st.warning("Enter a question about your table.")
    elif not header:
        st.warning("Enter at least one column name, separated by commas.")
    elif len(header) > MAX_COLS:
        st.warning(f"This model supports up to {MAX_COLS} columns per table.")
    else:
        source_text = encode_source(question, header)
        source_ids = tokenizer.encode(source_text) + [EOS_ID]
        if len(source_ids) > 512:
            st.warning("The question and table columns exceed the model's 512-token input limit.")
            st.stop()
        source = torch.tensor(source_ids, dtype=torch.long, device=device)

        with st.spinner("Generating SQL…"):
            if decoding == "Greedy":
                generated_ids = greedy_decode(model, source)
            else:
                generated_ids = beam_decode(model, source, beam_size=4)

        generated_text = tokenizer.decode(generated_ids)
        query = parse_sql(generated_text)
        if query is None:
            st.warning("The model could not produce a valid SQL form for this input.")
            st.code(generated_text or "(empty output)", language="text")
        elif query["sel"] >= len(header) or any(col >= len(header) for col, _, _ in query["conds"]):
            st.warning("The generated query refers to a column outside the supplied table.")
            st.code(generated_text, language="text")
        else:
            # Escape apostrophes for a valid SQL string literal in the display.
            readable_query = copy.deepcopy(query)
            readable_query["conds"] = [
                [col, op, value.replace("'", "''")]
                for col, op, value in readable_query["conds"]
            ]
            st.subheader("Generated SQL")
            st.code(to_readable_sql(readable_query, header), language="sql")
            with st.expander("Model output"):
                st.code(generated_text, language="text")
