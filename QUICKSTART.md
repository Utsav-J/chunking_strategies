# Quick Start Guide - Chunking Strategies Comparison Tool

## 🚀 Getting Started

### 1. Install Dependencies

If you haven't already, sync the project dependencies:

```bash
uv sync
```

This will install all required packages including:
- streamlit
- langchain
- llama-index
- sentence-transformers
- plotly
- pandas
- and more...

### 2. Run the Streamlit App

From the project root directory:

```bash
streamlit run streamlit_app/app.py
```

**OR** navigate to the streamlit_app folder:

```bash
cd streamlit_app
streamlit run app.py
```

### 3. Using the App

1. **Choose Input Method** (in sidebar):
   - Upload a text file (.txt, .md, .py)
   - OR paste text directly

2. **Select Chunking Strategies**:
   - Character Chunking
   - Recursive Character Chunking
   - Markdown Chunking
   - Python Chunking
   - Semantic Chunking
   - Cluster Semantic Chunking

3. **Configure Parameters**:
   - Adjust chunk sizes
   - Set overlap values
   - Fine-tune semantic thresholds

4. **Run & Compare**:
   - Click "Run Selected Strategies"
   - View comparison charts
   - Explore detailed chunk views
   - Download results

## 📝 Example Workflow

### For Text Documents:
1. Upload a `.txt` or `.md` file
2. Select "Recursive Character Chunking" and "Semantic Chunking"
3. Set chunk_size to 500, overlap to 50
4. Run and compare the results

### For Code Files:
1. Upload a `.py` file
2. Select "Python Chunking"
3. Set chunk_size to 300, overlap to 30
4. View how code structure is preserved

### For Semantic Analysis:
1. Upload a long-form text
2. Select "Semantic Chunking" with default settings
3. Adjust breakpoint_percentile (90-99)
4. Compare semantic boundaries

## 🎯 Key Features

- **Side-by-Side Comparison**: See how different strategies split your text
- **Visual Analytics**: Charts showing chunk counts and size distributions
- **Configurable**: Adjust all parameters for each strategy
- **Export**: Download chunked results
- **Multiple Inputs**: Support for .txt, .md, .py files

## ⚠️ Important Notes

- **First Run**: Semantic chunking will download the embedding model (~400MB)
- **Processing Time**: Semantic methods take longer but provide better quality
- **Memory**: Large files may require more RAM for semantic chunking
- **Browser**: Works best in Chrome, Firefox, or Edge

## 🐛 Troubleshooting

### "Module not found" errors:
```bash
uv sync
```

### "Streamlit not found":
```bash
pip install streamlit
```

### Semantic chunking not working:
Ensure numpy, scikit-learn, and sentence-transformers are installed:
```bash
pip install numpy scikit-learn sentence-transformers
```

## 📚 Next Steps

- Read the full documentation in `streamlit_app/README.md`
- Explore the individual chunking strategies in folders 1-5
- Try different parameters to optimize for your use case
- Compare results on your specific documents

## 💡 Tips

1. Start with simple strategies (Character, Recursive) for quick results
2. Use Semantic chunking for topic-based content
3. Document-specific chunkers (Markdown, Python) preserve structure better
4. Adjust parameters based on your downstream task (RAG, indexing, etc.)

