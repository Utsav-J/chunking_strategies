## Overview

Level 3 focuses on **Document-Specific Text Splitting**. This approach moves beyond general prose splitting (like in Levels 1 and 2) by leveraging the **physical structure** of specialized document formats such as code, configuration files, and rich media documents (like PDFs containing tables and images). By recognizing special characters and formatting conventions, Level 3 strategies can infer more about the document structure and make smarter splits.

## Concept: Leveraging Document Structure

The strategy behind Level 3 is to take advantage of formatters and structural elements common in specific document types (e.g., code formatters). This allows the splitter to group items that are semantically similar by nature of their format.

## Implementation Examples

### 1. Markdown Splitting

Markdown documents use specific symbols, like the pound symbol (`#`), to denote headers. Headers usually denote the topic being discussed, making them excellent break points for chunking.

*   **Separator Strategy:** Markdown splitting uses a sophisticated list of separators, including a new line followed by a header symbol (`#`) repeated between one and six times (H1 through H6).
*   **Result:** The split occurs at these header markers, ensuring that content under a single header (which is likely to be about the same topic) is grouped together.
*   **LangChain Implementation:** LangChain provides specialized splitters for different languages, and the specific separators used for Markdown can be inspected on their GitHub repository.

**Example:**
When splitting markdown text like "Fun in California" (H1) followed by an H2 header for "Driving," the splitter uses these headers to define the chunk boundaries.

### 2. Code Splitting (Python and JavaScript)

For code documents, the goal is to split based on logical programming structures (like classes or functions) to keep related code blocks together.

*   **Python:** LangChain’s Python Splitters split on structures like **classes**, **functions**, **indenting functions** (methods within a class), double new lines, new lines, spaces, and characters.
    *   **Result:** This allows an entire class definition, for example, to be enveloped within one document chunk.
*   **JavaScript (and other languages):** You can use the `RecursiveCharacterTextSplitter.from_language` function in LangChain, specifying the desired language (e.g., `Language.JS`).

### 3. Multimodal Splitting (PDFs, Tables, and Images)

Many older industries still rely on information stored in PDFs, which often contain complex elements like tables, pictures, and graphs. Level 3 extends chunking to handle these multimodal elements.

#### Handling Tables in PDFs

When chunking PDFs, you don't just split text; you want to pull out all elements, including tables.

*   **Tool:** The **Unstructured** library (`unstructured.io`) is recommended for parsing complicated or messy data types like millions of PDFs, using functions like `partition_pdf`.
*   **Strategy:** While tables are easy for humans to read, they are less straightforward for Language Models (LLMs). Since LLMs are often trained on **HTML tables** (and also markdown), the strategy is to extract the table data from the PDF and convert it into HTML format.
*   **Result:** The HTML representation of the table is what is passed to the LLM, allowing the model to make better sense of the structured data.

#### Handling Images in PDFs

Images need a special approach for retrieval because embedding models for text and images generally do not align, making similarity search difficult (though models like **CLIP** exist).

*   **Extraction:** Unstructured is used again with `partition_pdf` and the parameter `extract_images_in_pdf=True` to pull out images from the document and save them separately.
*   **Strategy: Text Summarization:** The proposed method (as a current practical solution) is to **generate a text summary of each image** using a multimodal LLM (like GPT-4 Vision Preview).
*   **Retrieval Process:**
    1.  The text summary is embedded and used for semantic search.
    2.  If the summary is returned during search, the retrieved summary can be used on its own, or the original image can be passed to a multimodal LLM along with the text query.
    *   **Example:** An image of baked goods arranged like continents resulted in a descriptive summary that could be used for retrieval.
