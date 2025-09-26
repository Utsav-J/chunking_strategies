# 📖 Chunking Techniques in NLP and RAG

Chunking is the process of splitting large text into smaller, manageable pieces (chunks) that can be stored, searched, or processed efficiently.  
It’s especially important in **Retrieval-Augmented Generation (RAG)** and **document indexing**, since most LLMs have token limits.

---

## 1. Fixed-Size Chunking
- **Definition**: Break text into equal-sized chunks (by characters, words, or tokens).
- **Example**: Split every 500 tokens.
- **Pros**: Simple, fast, predictable.
- **Cons**: May cut sentences or paragraphs mid-way, losing semantic meaning.

```text
[Chunk 1: tokens 0–499]
[Chunk 2: tokens 500–999]
...
````

---

## 2. Overlapping Sliding Window

* **Definition**: Fixed-size chunks, but with overlap to preserve context across boundaries.
* **Example**: 500-token chunks with 100-token overlap.
* **Pros**: Reduces risk of missing context at chunk boundaries.
* **Cons**: Increases storage and retrieval costs.

```text
[Chunk 1: tokens 0–499]
[Chunk 2: tokens 400–899]  (100-token overlap)
[Chunk 3: tokens 800–1299]
```

---

## 3. Sentence-Based Chunking

* **Definition**: Use sentence boundaries to define chunks.
* **Example**: Group 3–5 sentences per chunk.
* **Pros**: Preserves natural meaning.
* **Cons**: Chunk size can vary widely; may not fit token limits.

```text
[Chunk 1: Sentence 1–5]
[Chunk 2: Sentence 6–10]
```

---

## 4. Paragraph-Based Chunking

* **Definition**: Split text by paragraphs.
* **Pros**: Maintains semantic grouping.
* **Cons**: Some paragraphs may be too long; others too short.

```text
[Chunk 1: Paragraph 1]
[Chunk 2: Paragraph 2]
```

---

## 5. Recursive Chunking (Hybrid)

* **Definition**: Start with large blocks, then break them down recursively if they exceed a threshold.
* **Example**: Split by sections → paragraphs → sentences → words.
* **Pros**: Balances semantic structure and token constraints.
* **Cons**: More complex to implement.

---

## 6. Semantic Chunking (Embedding-Aware)

* **Definition**: Use embeddings or topic segmentation to find natural breakpoints (topic shifts, headings).
* **Pros**: Best semantic preservation; reduces irrelevant splits.
* **Cons**: Computationally expensive.

---

## 📊 Comparison Table

| Technique          | Preserves Meaning | Simplicity | Efficiency | Use Case             |
| ------------------ | ----------------- | ---------- | ---------- | -------------------- |
| Fixed-Size         | ❌                 | ✅          | ✅          | Large raw text       |
| Overlapping Window | ➖                 | ✅          | ➖          | Legal docs, research |
| Sentence-Based     | ✅                 | ✅          | ✅          | Conversational text  |
| Paragraph-Based    | ✅                 | ✅          | ➖          | Articles, reports    |
| Recursive Chunking | ✅                 | ➖          | ➖          | Mixed documents      |
| Semantic Chunking  | ✅✅                | ❌          | ❌          | Knowledge bases      |

---

## ✅ Best Practices

1. Choose chunking based on **document type** and **LLM token window**.
2. Always balance **chunk size** (too small = noisy retrieval, too large = token overflow).
3. Consider **overlaps** for context-heavy use cases.
4. For production RAG: **Recursive chunking + overlap** is often the sweet spot.

