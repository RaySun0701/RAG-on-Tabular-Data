# Two retrieval methods

------

## **Method 1: BM25 + Semantic Retrieval (Dense Retrieval)**

Best suited when your tables contain rich textual information.

### **Steps**

1. **Text Flattening (JSON Preprocessing)**:

   - Convert JSON tables into **searchable text representations**.

   - For example, concatenate headers and values into a format like:

     ```
     "Column1: value1, Column2: value2, ..."
     ```

2. **Traditional Retrieval (BM25)**:

   - Use BM25 to retrieve tables based on lexical similarity to the claim.

3. **Semantic Retrieval (Dense Retrieval)**:

   - Compute embeddings for both **claims** and **tables** using models like **BGE, MiniLM, or Contriever**.
   - Use **cosine similarity (FAISS)** to find the most relevant table.

4. **Hybrid Ranking (Score Fusion)**:

   - Combine BM25 and dense retrieval scores using a weighted sum: $\text{Final Score} = \alpha \times \text{BM25 Score} + (1 - \alpha) \times \text{Dense Score}$
   - The weight $\alpha$ can be fine-tuned based on experiments.

------

## **Method 2: RAG (Retrieval-Augmented Generation) for Table-Based Retrieval**

If your claims require reasoning across multiple tables, **RAG-based retrieval** can be used:

1. **Index Table Data**:

   - Store table embeddings using **BM25 + Dense Retrieval (FAISS-based index)**.

2. **Table Retrieval**:

   - Given a claim, retrieve the top **K** most relevant tables.

3. **LLM Fusion for Reasoning**:

   - Use an **LLM (GPT, Claude, etc.)** to aggregate information from the retrieved tables.

   - Example prompt:

     ```
     "Based on the following tables, answer the claim."
     ```

------

## **Method 3: Structured Retrieval for Tables**

Best suited for cases where table contents are well-structured with fixed fields.

1. **Indexing Strategy**:
   - Generate embeddings at different levels:
     - **Column embeddings** (per column name)
     - **Row embeddings** (per row content)
     - **Table embeddings** (entire table)
2. **Retrieval Strategy**:
   - First, **match column names** with the claim.
   - Then, use **dense retrieval** to rank the most relevant table portions.
3. **Re-ranking with a Fine-Tuned Model**:
   - Use **ColBERT or BERT-Reranker** to refine table ranking.

------

## **Recommended Approach**

If you want:

- **A quick, effective solution** → **Method 1 (BM25 + Dense Retrieval)**
- **More powerful reasoning capabilities** → **Method 2 (RAG)**
- **Highly structured retrieval** → **Method 3 (Structured Retrieval)**









The goal is **only to retrieve the most relevant table** based on a claim, the best approach would be **BM25 + Dense Retrieval (Hybrid Retrieval)**. 

------

### **Solution: Hybrid Table Retrieval**

1. **Convert JSON tables into text format** for better retrieval.

2. Index the tables

    using:

   - **BM25** (for keyword-based retrieval).
   - **Dense Retrieval** (for semantic matching using embeddings).

3. **Retrieve the most relevant table** by combining BM25 and semantic similarity scores.

------

### **Step 1: Preprocessing JSON Tables**

Convert JSON tables into a structured text format. A simple way is:

```python
import json

def json_to_text(json_str):
    """
    Convert a JSON table to a structured text format.
    """
    data = json.loads(json_str)
    text_representation = []
    
    # If it's a list of dictionaries (structured table format)
    if isinstance(data, list):
        headers = data[0].keys()  # Get column names
        text_representation.append(", ".join(headers))  # Column headers
        
        for row in data:
            values = [str(row[col]) for col in headers]
            text_representation.append(", ".join(values))  # Row values
            
    return " | ".join(text_representation)  # Final text format
```

This will turn a JSON table into a **searchable text string**.

------

### **Step 2: Indexing Tables (BM25 + Dense Embedding)**

1. Use **BM25** for keyword-based ranking.
2. Use **Dense Retrieval (FAISS + Embeddings)** for semantic similarity.

#### **1. BM25 Indexing**

```python
from rank_bm25 import BM25Okapi

# Convert JSON tables to text
table_texts = [json_to_text(table_json) for table_json in list_of_json_tables]

# Tokenize for BM25
tokenized_texts = [text.split() for text in table_texts]
bm25 = BM25Okapi(tokenized_texts)
```

#### **2. Dense Embedding Indexing (FAISS)**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en")

# Compute table embeddings
table_embeddings = model.encode(table_texts, normalize_embeddings=True)

# Build FAISS index
index = faiss.IndexFlatIP(table_embeddings.shape[1])  # Inner product for cosine similarity
index.add(table_embeddings)
```

------

### **Step 3: Querying the Most Relevant Table**

Now, given a **claim**, we retrieve the most relevant table.

#### **1. BM25 Retrieval**

```python
def retrieve_bm25_top_k(claim, k=5):
    tokenized_claim = claim.split()
    scores = bm25.get_scores(tokenized_claim)
    top_k_indices = np.argsort(scores)[-k:][::-1]  # Top-K highest scores
    return top_k_indices, scores[top_k_indices]
```

#### **2. Dense Retrieval (Semantic Matching)**

```python
def retrieve_dense_top_k(claim, k=5):
    claim_embedding = model.encode([claim], normalize_embeddings=True)
    _, top_k_indices = index.search(claim_embedding, k)
    return top_k_indices[0]  # FAISS returns 2D array
```

#### **3. Combine BM25 + Dense Scores**

A simple weighted combination:

```python
def hybrid_retrieve(claim, k=5, alpha=0.5):
    bm25_indices, bm25_scores = retrieve_bm25_top_k(claim, k)
    dense_indices = retrieve_dense_top_k(claim, k)
    
    # Convert indices to sets for intersection
    combined_scores = {}
    
    for idx, score in zip(bm25_indices, bm25_scores):
        combined_scores[idx] = alpha * score  # BM25 weighted score
        
    for idx in dense_indices:
        combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha)  # Dense score
    
    # Sort by highest combined score
    sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)
    
    return sorted_indices[0]  # Return the most relevant table index
```

------

### **Step 4: Retrieve and Display the Most Relevant Table**

```python
claim = "Find the table related to sales revenue trends."

# Get the best matching table index
best_table_idx = hybrid_retrieve(claim)

# Retrieve the actual JSON table
best_table_json = list_of_json_tables[best_table_idx]

print(best_table_json)  # This is the most relevant table
```

------

### **Why This Approach?**

- **BM25** → Handles **exact keyword matches** well.
- **Dense Retrieval** → Handles **semantic similarity** (even when words don’t match exactly).
- **Hybrid (BM25 + FAISS)** → Achieves **better accuracy** than using just one method.

