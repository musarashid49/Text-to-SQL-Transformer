# LinkedIn post draft

We built a Text-to-SQL model that turns a question about a table into a SQL query—using a Transformer implemented and trained from scratch in PyTorch.

For this project, our team trained a 7.58-million-parameter encoder–decoder Transformer on WikiSQL for 20 epochs using an NVIDIA T4 GPU. The model takes a natural-language question and the table’s column names, then generates a query using those columns.

On the held-out WikiSQL test set, beam search reached **64.59% execution accuracy** and **57.49% logical-form accuracy**. Execution accuracy measures whether the generated query returns the same answer as the reference query; logical-form accuracy requires the query structure itself to match. The results also show where the model still struggles, especially with selecting all the right WHERE conditions and values.

I also built a small Streamlit interface so people can try the trained model with their own question and column list.

**Project repository:** https://github.com/musarashid49/Text-to-SQL-Transformer

**Project write-up:** [Add the Medium blog link here, if publishing the blog]

#NLP #Transformers #PyTorch #MachineLearning #TextToSQL #GenerativeAI
