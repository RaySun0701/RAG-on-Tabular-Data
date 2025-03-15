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

The original TabFact data store claims and tables separately. We combine them together in JSON format.
```json
"2-15401676-3.html.csv": [
    [
        "haroldo be mention as a brazil scorer for 2 different game",
        "4 of the 5 game be for the south american championship",
        "..."
    ],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "1919 in brazilian football",
    {
        "columns": ["date", "result", "score", "brazil scorers", "competition"],
        "data": [
            ["may 11, 1919", "w", "6 - 0", "friedenreich (3), neco (2), haroldo", "south american championship"],
            ["..."]
        ]
    }
]
```

## Summary of Data Structure
- **Keys (file names)**: Represent different datasets.
- **Values**: Each dataset is stored as a list with:
  1. **Claims** (Claims about the dataset).
  2. **Binary Labels** (evaluating correctness of claims).
  3. **Topic** (topic for the data).
  4. **Tabular Data** (column names and row values).
