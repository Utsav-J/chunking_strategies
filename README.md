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

---

## 🚀 Interactive Comparison Tool

This project now includes a **Streamlit-based interactive comparison tool** to visualize and compare all chunking strategies side-by-side!

### Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app/app.py
   ```

3. **Use the app:**
   - Upload a text file or paste your content
   - Select multiple chunking strategies
   - Adjust parameters for each strategy
   - Compare results side-by-side with visualizations
   - Download chunked results

### Features

- 📊 **Visual Comparisons**: Charts showing number of chunks and size distributions
- ⚙️ **Configurable**: Adjust parameters for each strategy
- 📝 **Detailed Views**: Preview individual chunks from each strategy
- 📥 **Export**: Download chunked results as text files
- 🔄 **Multiple Strategies**: Run and compare up to 6 different chunking methods

See `QUICKSTART.md` for detailed instructions and `streamlit_app/README.md` for full documentation.

---

## 📁 Project Structure

```
chunking-strategies/
├── 1-character-chunking/       # Fixed-size character chunking
├── 2-recursive-character/      # Recursive character text splitter
├── 3-document-specific/        # Document-type aware chunkers
│   ├── markdown.py
│   ├── python_splitter.py
│   └── language_splitter.py
├── 4-semantic-chunking/        # Embedding-based semantic chunking
├── 5-cluster-semantic-chunking/ # Global optimization semantic chunking
├── streamlit_app/              # 🆕 Interactive comparison tool
│   ├── app.py                  # Main Streamlit application
│   ├── unified_chunkers.py     # Unified chunking interface
│   ├── README.md               # App documentation
│   └── sample_text.txt         # Example text file
├── pyproject.toml              # Dependencies
├── QUICKSTART.md               # 🆕 Quick start guide
└── README.md                   # This file

