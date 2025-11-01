# Streamlit App Utilities

This directory contains modular components for the chunking strategies comparison Streamlit app.

## Module Structure

### `visualization.py`
Handles all visualization and display logic for chunking results.

**Functions:**
- `display_chunk_comparison(results)` - Shows side-by-side comparison of strategies
- `display_chunks_detailed(strategy_name, result)` - Shows detailed view of a single strategy

**Exports:**
- Chunk comparison metrics and charts
- Distribution histograms
- Chunk previews and statistics
- Download functionality

### `chunking.py`
Contains the chunking execution logic.

**Functions:**
- `run_chunking_strategy(chunker, strategy_name, text, params)` - Executes a single strategy
- `process_strategies(chunker, selected_strategies, text_content, strategy_params)` - Processes all selected strategies with progress tracking

**Exports:**
- Strategy execution logic
- Progress tracking
- Error handling

### `ui_components.py`
All UI components and user interface elements.

**Functions:**
- `render_sidebar(strategies_info)` - Renders the configuration sidebar
- `render_parameter_input(param, default_value, strategy_name)` - Renders parameter inputs
- `render_text_input()` - Renders file upload/text input section
- `render_welcome_message()` - Shows welcome message
- `render_footer()` - Shows app footer

**Exports:**
- Sidebar with strategy selection and parameters
- Text input components
- UI helpers

### `session_state.py`
Manages Streamlit session state.

**Functions:**
- `initialize_session_state()` - Initializes session state variables
- `get_chunker()` - Gets or creates the chunker instance

**Exports:**
- Session state management
- Chunker singleton pattern

## Usage

All modules are imported in `app.py` and used to compose the main application:

```python
from utils.visualization import display_chunk_comparison, display_chunks_detailed
from utils.chunking import process_strategies
from utils.ui_components import render_sidebar, render_text_input
from utils.session_state import initialize_session_state, get_chunker
```

## Benefits of Modular Design

1. **Separation of Concerns** - Each module has a single responsibility
2. **Maintainability** - Easier to locate and fix bugs
3. **Reusability** - Components can be reused in other contexts
4. **Testability** - Individual functions can be tested in isolation
5. **Readability** - Main app.py is now much cleaner and easier to understand

