"""
Streamlit app for comparing different chunking strategies.
Allows users to upload text files and visualize chunking results.
"""
import streamlit as st

# Import utility modules
from utils.visualization import display_unified_insights
from utils.chunking import process_strategies
from utils.ui_components import (
    render_sidebar, 
    render_text_input, 
    render_welcome_message, 
    render_footer
)
from utils.session_state import initialize_session_state, get_chunker
from unified_chunkers import get_available_strategies

# Page config
st.set_page_config(
    page_title="Chunking Strategies Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

# Get chunker and strategies
chunker = get_chunker()
strategies_info = get_available_strategies()


# ============ Main App ============

st.title("📖 Chunking Strategies Comparison Tool")
st.markdown("""
Compare different text chunking strategies side-by-side. Upload a text file or paste your content,
then select and run multiple chunking strategies to see how they differ.
""")

# Render sidebar and get user selections
selected_strategies, strategy_params, run_button = render_sidebar(strategies_info)

# Main content area - Input section
text_content = render_text_input()

# Check if content is ready
if not text_content:
    render_welcome_message()
    st.stop()

# Process strategies when button is clicked
if run_button and selected_strategies:
    results = process_strategies(chunker, selected_strategies, text_content, strategy_params)
    st.session_state.results = results

elif run_button and not selected_strategies:
    st.warning("⚠️ Please select at least one strategy to run.")

# Display results in unified insights section
if st.session_state.results:
    display_unified_insights(st.session_state.results)

# Footer
render_footer()
