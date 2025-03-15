# **RAG-on-Tabular-Data**

# Abstract

Large Language Models (LLMs) have demonstrated remarkable performance in natural language processing (NLP) tasks but often suffer from hallucinations when verifying claims against structured tabular data. This limitation arises due to their reliance on pre-trained knowledge, which lacks explicit retrieval mechanisms. In this study, we propose a Retrieval-Augmented Generation (RAG) pipeline to mitigate hallucinations by incorporating relevant tabular data before reasoning. We leverage the TabFact dataset, a large-scale benchmark for table-based fact verification, to evaluate the effectiveness of our approach. Our system retrieves relevant tables from a structured database, augments LLM input with retrieved data, and assesses the reduction in hallucination rates compared to baseline models. Preliminary results suggest that RAG significantly enhances fact verification accuracy by reducing false information generation.

# Data Description

The TabFact dataset is a large-scale benchmark for table-based fact verification, containing:

- 16,000 tables and 118,000 claims.
- Each claim is labeled as either:
  - **Entailed** (factually correct based on the table).
  - **Refuted** (factually incorrect based on the table).

Tables consist of structured numerical and categorical data, requiring multi-row and multi-column reasoning.
