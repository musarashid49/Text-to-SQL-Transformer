# LinkedIn post draft

My teammate [**Areesha Saqib**](https://www.linkedin.com/in/areesha-saqib-a55152429/) and I built an end-to-end Text-to-SQL Transformer from scratch in PyTorch: given an English question and a table’s column names, the model generates a WikiSQL query.

No pre-trained checkpoints. No high-level Transformer modules. The goal was to implement the encoder–decoder architecture ourselves and learn to map questions to columns, operators, aggregations, and values using the WikiSQL dataset.

**The Problem: Turning Questions into Structured Queries**

WikiSQL questions refer to columns and values in tables the model has not seen before. A useful system must select the right column, choose an aggregation when needed, and construct the correct WHERE conditions—not just produce text that looks like SQL.

**The Architecture (7.58M Trainable Parameters):**

**1- Scaled Dot-Product Attention:** Implemented masked attention and four parallel attention heads using basic PyTorch layers.

**2- Encoder–Decoder Transformer:** Built 3 encoder and 3 decoder layers with 256-dimensional states, 1,024-dimensional feed-forward layers, dropout 0.1, and the paper’s post-norm residual structure.

**3- Schema-Aware Inputs:** Used the provided input-preparation layer and special column markers (`<c0>`, `<c1>`, …) so the model points to columns instead of generating their names.

**4- Shared Weights:** Tied the encoder and decoder token embeddings to the output projection, with a shared 8,000-piece BPE vocabulary.

**Training Dynamics & Optimisation**

The model was trained from random initialisation for 20 epochs on an NVIDIA T4 GPU (batch size 64). Training used Adam (β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹), 0.1 label smoothing, and the Transformer learning-rate schedule with 4,000 warm-up steps.

Training loss decreased from **4.35 to 1.36**, while dev loss decreased from **3.15 to 1.48**. The run took about **21.5 minutes**, and the checkpoint with the lowest dev loss was selected.

**Evaluation Results:**

**WikiSQL Dev (8,421 examples):** Greedy: **57.00% logical-form accuracy** | **64.79% execution accuracy**

**WikiSQL Dev, beam size 4:** **57.40% logical-form accuracy** | **65.24% execution accuracy**

**WikiSQL Test (15,878 examples), beam size 4:** **57.49% logical-form accuracy** | **64.59% execution accuracy**

Execution accuracy checks whether the generated query returns the same result as the reference query. The results also make the remaining challenge clear: getting every selected column, condition, and value right.

Code: [**https://github.com/musarashid49/Text-to-SQL-Transformer**](https://github.com/musarashid49/Text-to-SQL-Transformer)

Technical write-up: [**Add the Medium blog link here**]

#NLP #Transformers #PyTorch #MachineLearning #TextToSQL #GenerativeAI
