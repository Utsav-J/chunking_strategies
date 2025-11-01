# Chunking Strategies Comparison Tool

A Streamlit-based interactive tool for comparing different text chunking strategies side-by-side.

## 🚀 Quick Start

### Prerequisites

Install dependencies using uv:

```bash
uv sync
```

### Running the App

From the project root directory:

```bash
streamlit run streamlit_app/app.py
```

Or navigate to the streamlit_app directory:

```bash
cd streamlit_app
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## 📖 Features

### 1. **Multiple Input Methods**
   - Upload text files (.txt, .md, .py)
   - Paste text directly into the app

### 2. **Six Chunking Strategies**
   - **Character Chunking**: Fixed-size character splitting
   - **Recursive Character Chunking**: Hybrid approach with multiple separators
   - **Markdown Chunking**: Markdown-aware splitting
   - **Python Chunking**: Python code-aware splitting
   - **Semantic Chunking**: Embedding-based semantic splitting
   - **Cluster Semantic Chunking**: Global optimization for semantic coherence

### 3. **Configurable Parameters**
   - Adjust chunk sizes and overlaps
   - Fine-tune semantic thresholds
   - Set breakpoint percentiles

### 4. **Visual Comparisons**
   - Side-by-side metrics comparison
   - Charts showing number of chunks and sizes
   - Chunk size distribution histograms

### 5. **Detailed Views**
   - Preview individual chunks
   - View strategy metadata
   - Download chunked results

## 🎯 Use Cases

- **RAG System Development**: Test which chunking strategy works best for your documents
- **Document Processing**: Compare chunk sizes and overlaps across strategies
- **Research**: Analyze semantic vs. rule-based chunking approaches
- **Education**: Learn how different strategies affect text splitting

## 💡 Tips

1. Start with a small text sample to understand strategy differences
2. Adjust parameters based on your document type (code, prose, markdown)
3. Use the comparison view to identify optimal strategy for your use case
4. Semantic chunking works best for documents with clear topic boundaries
5. Document-specific chunkers (Markdown, Python) preserve structure better

## 📊 Understanding the Visualizations

### Comparison View
- **Number of Chunks**: How many pieces the text was split into
- **Average Chunk Size**: Mean characters per chunk
- **Distribution**: Visual representation of chunk size variability

### Detailed View
- **Chunk Preview**: First 5 chunks from each strategy
- **Statistics**: Min, max, median, and standard deviation of chunk sizes
- **Export**: Download all chunks as a text file

## 🔧 Architecture

- `app.py`: Main Streamlit application with UI and visualization
- `unified_chunkers.py`: Unified interface wrapping all chunking implementations

## 🐛 Troubleshooting

### Semantic Chunking Not Available
If semantic chunking shows as unavailable, ensure you have:
- numpy
- scikit-learn
- sentence-transformers

### Slow Performance
- Semantic chunking is computationally expensive
- Consider using smaller text samples for testing
- Reduce buffer size and adjust percentile thresholds

## 📝 Notes

- First run of semantic strategies will download the embedding model
- Large files may take time to process with semantic methods
- Results are stored in session state and persist during your session

