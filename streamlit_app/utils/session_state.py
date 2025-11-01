"""
Session state management utilities.
"""
import streamlit as st
from unified_chunkers import UnifiedChunker


def initialize_session_state():
    """Initialize session state variables"""
    
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    if 'chunker' not in st.session_state:
        st.session_state.chunker = None


def get_chunker() -> UnifiedChunker:
    """Get or create the chunker instance"""
    
    if st.session_state.chunker is None:
        st.session_state.chunker = UnifiedChunker()
    
    return st.session_state.chunker

