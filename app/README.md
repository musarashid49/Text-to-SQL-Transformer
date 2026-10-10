# Text-to-SQL web app

This Streamlit app loads the trained checkpoint and its matching SentencePiece model,
accepts a question and comma-separated table columns, then displays generated SQL using
the supplied column names. It supports greedy decoding and beam search with width 4.

From the repository root, install requirements and start the app:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app expects `results/inference.pt` and `starter/sql_sp.model`, exported from the same
training run. It uses CUDA when available, then Apple MPS, then CPU.
