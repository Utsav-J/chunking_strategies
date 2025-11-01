"""
Visualization components for displaying chunking results.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict
from unified_chunkers import ChunkingResult


def display_unified_insights(results: Dict[str, ChunkingResult]):
    """Display unified insights section with comparison metrics and sample chunks"""
    
    if not results:
        st.info("No chunking results to display yet. Upload a file and run chunking to see insights.")
        return
    
    # ========== Overview Metrics ==========
    st.subheader("📊 Overview Metrics")
    
    # Prepare data for comparison
    comparison_data = []
    for strategy_name, result in results.items():
        comparison_data.append({
            "Strategy": strategy_name,
            "Number of Chunks": result.num_chunks,
            "Avg Chunk Size (chars)": int(result.avg_chunk_size),
            "Total Characters": sum(len(c) for c in result.chunks),
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Display metrics
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
    
    # Download all button
    st.markdown("### 📥 Export All Chunks")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Download all chunked results from all strategies in a single file")
    with col2:
        # Prepare combined download
        all_chunks_text = ""
        for strategy_name, result in results.items():
            all_chunks_text += f"\n\n{'='*80}\n"
            all_chunks_text += f"STRATEGY: {strategy_name}\n"
            all_chunks_text += f"{'='*80}\n\n"
            all_chunks_text += "\n\n---CHUNK_DELIMITER---\n\n".join(result.chunks)
            all_chunks_text += "\n\n"
        
        st.download_button(
            label="📥 Download All",
            data=all_chunks_text,
            file_name="all_chunking_results.txt",
            mime="text/plain",
            key="download_all_strategies"
        )
    
    st.divider()
    
    # ========== Visualizations ==========
    st.subheader("📈 Visualizations")
    
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
    
    st.divider()
    
    # ========== Sample Chunks Section ==========
    st.subheader("🔍 Sample Chunks Comparison")
    
    # Get all strategies for comparison
    sample_chunks_data = {
        "First Chunk": {},
        "Last Chunk": {},
        "Largest Chunk": {},
        "Smallest Chunk": {}
    }
    
    for strategy_name, result in results.items():
        chunks = result.chunks
        if chunks:
            sample_chunks_data["First Chunk"][strategy_name] = chunks[0]
            sample_chunks_data["Last Chunk"][strategy_name] = chunks[-1]
            sample_chunks_data["Largest Chunk"][strategy_name] = max(chunks, key=len)
            sample_chunks_data["Smallest Chunk"][strategy_name] = min(chunks, key=len)
    
    # Display sample chunks in a tabbed format
    tabs = st.tabs(list(sample_chunks_data.keys()))
    
    for idx, (sample_type, strategies_chunks) in enumerate(sample_chunks_data.items()):
        with tabs[idx]:
            if not strategies_chunks:
                st.info("No chunks available to display.")
                continue
            
            # Create expandable sections for each strategy
            for strategy_name, chunk_content in strategies_chunks.items():
                chunk_size = len(chunk_content)
                
                with st.expander(f"{strategy_name} ({chunk_size} chars)", expanded=True):
                    # Display truncated chunk with option to see full content
                    max_display_length = 500
                    if len(chunk_content) > max_display_length:
                        st.text(chunk_content[:max_display_length])
                        st.text("...")
                        st.text(f"[Content truncated - Full chunk is {chunk_size} characters]")
                        with st.expander("Show full chunk"):
                            st.text(chunk_content)
                    else:
                        st.text(chunk_content)
    
    st.divider()
    
    # ========== Detailed Statistics by Strategy ==========
    st.subheader("📊 Detailed Statistics")
    
    # Create expandable sections for each strategy's detailed stats
    for strategy_name, result in results.items():
        with st.expander(f"📋 {strategy_name}", expanded=False):
            col1, col2 = st.columns(2)
            
            chunk_sizes = [len(c) for c in result.chunks]
            
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
            
            # Metadata
            with st.expander("Strategy Information", expanded=False):
                metadata = result.metadata.copy()
                st.json(metadata)
            
            # Download button for this strategy
            st.markdown("### 📥 Export Chunks")
            chunk_text = "\n\n---CHUNK_DELIMITER---\n\n".join(result.chunks)
            st.download_button(
                label=f"Download {len(result.chunks)} chunks as TXT",
                data=chunk_text,
                file_name=f"{strategy_name.lower().replace(' ', '_')}_chunks.txt",
                mime="text/plain",
                key=f"download_{strategy_name}"
            )
    
    st.divider()
    
    # ========== Footer ==========
    st.markdown("💡 **Tip:** Scroll through the insights to compare how different strategies chunk your text!")
