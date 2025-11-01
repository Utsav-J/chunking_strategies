"""
Chunking logic and execution functions.
"""
import streamlit as st
from typing import Dict
from unified_chunkers import UnifiedChunker, ChunkingResult


def run_chunking_strategy(
    chunker: UnifiedChunker,
    strategy_name: str, 
    text: str, 
    params: Dict
) -> ChunkingResult:
    """Run a specific chunking strategy with given parameters"""
    
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


def process_strategies(
    chunker: UnifiedChunker,
    selected_strategies: list,
    text_content: str,
    strategy_params: Dict
) -> Dict[str, ChunkingResult]:
    """Process all selected strategies and return results"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    for i, strategy_name in enumerate(selected_strategies):
        status_text.text(f"Processing {strategy_name}...")
        try:
            result = run_chunking_strategy(chunker, strategy_name, text_content, strategy_params[strategy_name])
            results[strategy_name] = result
        except Exception as e:
            st.error(f"Failed to process {strategy_name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(selected_strategies))
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        st.success(f"✅ Successfully processed {len(results)} strategies!")
    
    return results

