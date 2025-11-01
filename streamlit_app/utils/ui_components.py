"""
UI components for the Streamlit app including sidebar and input sections.
"""
import streamlit as st
from typing import Dict, List, Tuple


def render_sidebar(strategies_info: Dict) -> Tuple[List[str], Dict, bool]:
    """
    Render the sidebar with strategy selection and parameters.
    
    Returns:
        Tuple of (selected_strategies, strategy_params, run_button_clicked)
    """
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Strategy selection
        st.subheader("🎯 Select Strategies")
        selected_strategies = []
        
        for strategy_name, info in strategies_info.items():
            available = info["available"]
            url = info.get("url", "#")
            if available:
                col1, col2 = st.columns([8, 1])
                with col1:
                    if st.checkbox(strategy_name, value=False, key=f"cb_{strategy_name}"):
                        selected_strategies.append(strategy_name)
                with col2:
                    # Info icon with link to strategy documentation
                    st.markdown(f'<a href="{url}" target="_blank" title="{info["description"]}"><span style="font-size: 1.2em;">⇗</span></a>', unsafe_allow_html=True)
            else:
                col1, col2 = st.columns([8, 1])
                with col1:
                    st.checkbox(strategy_name, value=False, disabled=True, key=f"cb_disabled_{strategy_name}")
                with col2:
                    st.markdown(f'<a href="{url}" target="_blank" title="{info["description"]}"><span style="font-size: 1.2em;">ℹ️</span></a>', unsafe_allow_html=True)
        
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
                strategy_params[strategy_name][param] = render_parameter_input(
                    param, 
                    defaults.get(param), 
                    strategy_name
                )
        
        st.divider()
        
        # Run button
        run_button = st.button("🚀 Run Selected Strategies", type="primary", use_container_width=True)
        
        # Clear results button
        if st.session_state.results:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.results = {}
                st.rerun()
        
        return selected_strategies, strategy_params, run_button


def render_parameter_input(param: str, default_value, strategy_name: str):
    """Render appropriate input widget for a parameter"""
    
    if param == "chunk_size":
        return st.number_input(
            f"{param.replace('_', ' ').title()}",
            min_value=10,
            max_value=5000,
            value=default_value if default_value else 200,
            key=f"{strategy_name}_{param}",
            step=10
        )
    elif param == "chunk_overlap":
        return st.number_input(
            f"{param.replace('_', ' ').title()}",
            min_value=0,
            max_value=500,
            value=default_value if default_value else 20,
            key=f"{strategy_name}_{param}",
            step=5
        )
    elif param == "buffer_size":
        return st.number_input(
            f"{param.replace('_', ' ').title()}",
            min_value=0,
            max_value=5,
            value=default_value if default_value else 1,
            key=f"{strategy_name}_{param}"
        )
    elif param == "breakpoint_percentile":
        return st.number_input(
            f"{param.replace('_', ' ').title()}",
            min_value=0.0,
            max_value=100.0,
            value=default_value if default_value else 95.0,
            key=f"{strategy_name}_{param}",
            step=1.0
        )
    elif param == "max_chunk_size":
        return st.number_input(
            f"{param.replace('_', ' ').title()}",
            min_value=50,
            max_value=2000,
            value=default_value if default_value else 400,
            key=f"{strategy_name}_{param}",
            step=50
        )
    elif param == "min_chunk_size":
        return st.number_input(
            f"{param.replace('_', ' ').title()}",
            min_value=10,
            max_value=200,
            value=default_value if default_value else 50,
            key=f"{strategy_name}_{param}",
            step=10
        )
    else:
        # Default input for unknown parameters
        return st.number_input(
            f"{param.replace('_', ' ').title()}",
            value=default_value if default_value else 0,
            key=f"{strategy_name}_{param}"
        )


def render_text_input() -> str:
    """Render the text input section and return the content"""
    
    st.subheader("📝 Input Text")
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Paste Text"],
        horizontal=True,
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
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Characters", f"{len(text_content):,}")
        with col2:
            st.metric("Words", f"{len(text_content.split()):,}")
    
    st.divider()
    
    return text_content


def render_welcome_message():
    """Render welcome message when no text is provided"""
    
    st.warning("👆 Please upload a file or paste text above to get started.")
    
    # Show strategy descriptions
    with st.expander("📚 Learn about the chunking strategies", expanded=True):
        st.markdown("""
        ### Available Chunking Strategies
        
        1. **Character Chunking**: Fixed-size character splitting (simple, fast)
        2. **Recursive Character Chunking**: Hybrid approach with multiple separators
        3. **Semantic Chunking**: Embedding-based semantic splitting
        4. **Cluster Semantic Chunking**: Global optimization for semantic coherence
        """)


def render_footer():
    """Render the app footer"""
    st.markdown("---")
    st.caption("💡 **Tip:** Try running multiple strategies on the same text to see how they differ!")

