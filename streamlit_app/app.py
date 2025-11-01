"""
Streamlit app for comparing different chunking strategies.
Allows users to upload text files and visualize chunking results.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from unified_chunkers import UnifiedChunker, get_available_strategies, ChunkingResult
import io
from typing import List, Dict

# Page config
st.set_page_config(
    page_title="Chunking Strategies Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'chunker' not in st.session_state:
    st.session_state.chunker = None

# Initialize chunker
if st.session_state.chunker is None:
    st.session_state.chunker = UnifiedChunker()

# Get available strategies
strategies_info = get_available_strategies()


def display_chunk_comparison(results: Dict[str, ChunkingResult]):
    """Display comparison metrics across different chunking strategies"""
    
    # Prepare data for comparison
    comparison_data = []
    for strategy_name, result in results.items():
        comparison_data.append({
            "Strategy": strategy_name,
            "Number of Chunks": result.num_chunks,
            "Avg Chunk Size (chars)": int(result.avg_chunk_size),
            "Total Characters": sum(len(c) for c in result.chunks),
        })
    
    if not comparison_data:
        st.info("No chunking results to compare yet. Upload a file and run chunking to see comparisons.")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.subheader("📊 Strategy Comparison")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Strategies Compared", len(df))
    with col2:
        avg_chunks = df["Number of Chunks"].mean()
        st.metric("Avg Chunks", f"{avg_chunks:.1f}")
    with col3:
        avg_size = df["Avg Chunk Size (chars)"].mean()
        st.metric("Avg Chunk Size", f"{avg_size:.0f} chars")
    with col4:
        total_chars = df["Total Characters"].iloc[0] if len(df) > 0 else 0
        st.metric("Total Characters", f"{total_chars:,}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Number of Chunks**")
        fig = px.bar(
            df, 
            x="Strategy", 
            y="Number of Chunks",
            color="Number of Chunks",
            color_continuous_scale="Blues"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Average Chunk Size**")
        fig = px.bar(
            df, 
            x="Strategy", 
            y="Avg Chunk Size (chars)",
            color="Avg Chunk Size (chars)",
            color_continuous_scale="Greens"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)


def display_chunks_detailed(strategy_name: str, result: ChunkingResult):
    """Display detailed chunk information for a specific strategy"""
    st.subheader(f"📝 {strategy_name} - Detailed View")
    
    # Metadata
    with st.expander("Strategy Information", expanded=False):
        metadata = result.metadata.copy()
        st.json(metadata)
    
    # Chunk size distribution
    chunk_sizes = [len(c) for c in result.chunks]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Chunk Size Distribution**")
        fig = px.histogram(
            x=chunk_sizes,
            nbins=20,
            labels={"x": "Chunk Size (characters)", "count": "Frequency"},
            color_discrete_sequence=["skyblue"]
        )
        fig.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Statistics**")
        st.write(f"- **Min size:** {min(chunk_sizes)} chars")
        st.write(f"- **Max size:** {max(chunk_sizes)} chars")
        st.write(f"- **Median size:** {pd.Series(chunk_sizes).median():.0f} chars")
        st.write(f"- **Std deviation:** {pd.Series(chunk_sizes).std():.0f} chars")
    
    # Chunk preview
    st.write(f"**Chunk Preview ({min(5, len(result.chunks))} of {len(result.chunks)})**")
    
    # Display first few chunks
    for i, chunk in enumerate(result.chunks[:5]):
        with st.expander(f"Chunk {i+1} ({len(chunk)} chars)", expanded=(i==0)):
            st.text(chunk)
    
    # Download chunks
    st.markdown("### 📥 Export Chunks")
    chunk_text = "\n\n---CHUNK_DELIMITER---\n\n".join(result.chunks)
    st.download_button(
        label=f"Download {len(result.chunks)} chunks as TXT",
        data=chunk_text,
        file_name=f"{strategy_name.lower().replace(' ', '_')}_chunks.txt",
        mime="text/plain"
    )


def run_chunking_strategy(
    strategy_name: str, 
    text: str, 
    params: Dict
) -> ChunkingResult:
    """Run a specific chunking strategy with given parameters"""
    chunker = st.session_state.chunker
    
    try:
        with st.spinner(f"Running {strategy_name}..."):
            if strategy_name == "Character Chunking":
                return chunker.character_chunking(
                    text, 
                    chunk_size=params.get("chunk_size", 100),
                    chunk_overlap=params.get("chunk_overlap", 0)
                )
            
            elif strategy_name == "Recursive Character Chunking":
                return chunker.recursive_character_chunking(
                    text,
                    chunk_size=params.get("chunk_size", 200),
                    chunk_overlap=params.get("chunk_overlap", 20)
                )
            
            elif strategy_name == "Markdown Chunking":
                return chunker.markdown_chunking(
                    text,
                    chunk_size=params.get("chunk_size", 200),
                    chunk_overlap=params.get("chunk_overlap", 25)
                )
            
            elif strategy_name == "Python Chunking":
                return chunker.python_chunking(
                    text,
                    chunk_size=params.get("chunk_size", 200),
                    chunk_overlap=params.get("chunk_overlap", 25)
                )
            
            elif strategy_name == "Semantic Chunking":
                return chunker.semantic_chunking(
                    text,
                    buffer_size=params.get("buffer_size", 1),
                    breakpoint_percentile=params.get("breakpoint_percentile", 95.0)
                )
            
            elif strategy_name == "Cluster Semantic Chunking":
                return chunker.cluster_semantic_chunking(
                    text,
                    max_chunk_size=params.get("max_chunk_size", 400),
                    min_chunk_size=params.get("min_chunk_size", 50)
                )
            
    except Exception as e:
        st.error(f"Error running {strategy_name}: {str(e)}")
        raise


# ============ Main App ============

st.title("📖 Chunking Strategies Comparison Tool")
st.markdown("""
Compare different text chunking strategies side-by-side. Upload a text file or paste your content,
then select and run multiple chunking strategies to see how they differ.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    st.subheader("📁 Input Source")
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Paste Text"],
        help="Upload a text file or paste text directly"
    )
    
    text_content = ""
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt', 'md', 'py'],
            help="Supported formats: .txt, .md, .py"
        )
        if uploaded_file:
            text_content = uploaded_file.read().decode('utf-8')
    
    else:  # Paste Text
        text_content = st.text_area(
            "Paste your text here:",
            height=200,
            help="Enter the text you want to chunk"
        )
    
    # Display file info
    if text_content:
        st.info(f"📄 **Characters:** {len(text_content):,}\n📊 **Words:** {len(text_content.split()):,}")
    
    st.divider()
    
    # Strategy selection
    st.subheader("🎯 Select Strategies")
    selected_strategies = []
    
    for strategy_name, info in strategies_info.items():
        available = info["available"]
        if available:
            if st.checkbox(strategy_name, value=False, help=info["description"]):
                selected_strategies.append(strategy_name)
        else:
            st.checkbox(strategy_name, value=False, disabled=True, 
                       help=f"Not available: {info['description']}")
    
    st.divider()
    
    # Strategy parameters
    st.subheader("🔧 Strategy Parameters")
    strategy_params = {}
    
    for strategy_name in selected_strategies:
        info = strategies_info[strategy_name]
        defaults = info["defaults"]
        params = info["params"]
        
        st.write(f"**{strategy_name}**")
        
        strategy_params[strategy_name] = {}
        for param in params:
            if param == "chunk_size":
                strategy_params[strategy_name][param] = st.number_input(
                    f"{param.replace('_', ' ').title()}",
                    min_value=10,
                    max_value=5000,
                    value=defaults.get(param, 200),
                    key=f"{strategy_name}_{param}",
                    step=10
                )
            elif param == "chunk_overlap":
                strategy_params[strategy_name][param] = st.number_input(
                    f"{param.replace('_', ' ').title()}",
                    min_value=0,
                    max_value=500,
                    value=defaults.get(param, 20),
                    key=f"{strategy_name}_{param}",
                    step=5
                )
            elif param == "buffer_size":
                strategy_params[strategy_name][param] = st.number_input(
                    f"{param.replace('_', ' ').title()}",
                    min_value=0,
                    max_value=5,
                    value=defaults.get(param, 1),
                    key=f"{strategy_name}_{param}"
                )
            elif param == "breakpoint_percentile":
                strategy_params[strategy_name][param] = st.number_input(
                    f"{param.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=100.0,
                    value=defaults.get(param, 95.0),
                    key=f"{strategy_name}_{param}",
                    step=1.0
                )
            elif param == "max_chunk_size":
                strategy_params[strategy_name][param] = st.number_input(
                    f"{param.replace('_', ' ').title()}",
                    min_value=50,
                    max_value=2000,
                    value=defaults.get(param, 400),
                    key=f"{strategy_name}_{param}",
                    step=50
                )
            elif param == "min_chunk_size":
                strategy_params[strategy_name][param] = st.number_input(
                    f"{param.replace('_', ' ').title()}",
                    min_value=10,
                    max_value=200,
                    value=defaults.get(param, 50),
                    key=f"{strategy_name}_{param}",
                    step=10
                )
    
    st.divider()
    
    # Run button
    run_button = st.button("🚀 Run Selected Strategies", type="primary", use_container_width=True)
    
    # Clear results button
    if st.session_state.results:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.results = {}
            st.rerun()

# Main content area
if not text_content:
    st.warning("👆 Please upload a file or paste text in the sidebar to get started.")
    
    # Show strategy descriptions
    with st.expander("📚 Learn about the chunking strategies", expanded=True):
        st.markdown("""
        ### Available Chunking Strategies
        
        1. **Character Chunking**: Fixed-size character splitting (simple, fast)
        2. **Recursive Character Chunking**: Hybrid approach with multiple separators
        3. **Markdown Chunking**: Markdown-aware splitting
        4. **Python Chunking**: Python code-aware splitting
        5. **Semantic Chunking**: Embedding-based semantic splitting
        6. **Cluster Semantic Chunking**: Global optimization for semantic coherence
        """)
    
    st.stop()

# Run strategies when button is clicked
if run_button and selected_strategies:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    for i, strategy_name in enumerate(selected_strategies):
        status_text.text(f"Processing {strategy_name}...")
        try:
            result = run_chunking_strategy(strategy_name, text_content, strategy_params[strategy_name])
            results[strategy_name] = result
        except Exception as e:
            st.error(f"Failed to process {strategy_name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(selected_strategies))
    
    st.session_state.results = results
    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Successfully processed {len(results)} strategies!")

elif run_button and not selected_strategies:
    st.warning("⚠️ Please select at least one strategy to run.")

# Display results
if st.session_state.results:
    # Tabs for different views
    tab1, tab2 = st.tabs(["📊 Comparison View", "📝 Detailed View"])
    
    with tab1:
        display_chunk_comparison(st.session_state.results)
    
    with tab2:
        # Strategy selector for detailed view
        strategy_for_detail = st.selectbox(
            "Select a strategy to view in detail:",
            list(st.session_state.results.keys())
        )
        display_chunks_detailed(strategy_for_detail, st.session_state.results[strategy_for_detail])

# Footer
st.markdown("---")
st.caption("💡 **Tip:** Try running multiple strategies on the same text to see how they differ!")

