## 1. Overview of Text Splitting

Text splitting, or chunking, is the **Art and Science of splitting your large data into smaller pieces**. This practice is foundational for language model practitioners. The goal of splitting is to make the data optimal for your specific task and language model, ensuring the model is given **only the information that it needs** and nothing more.

The ultimate objective (the **Chunking Commandment**) is **not to chunk for chunking sake**. Instead, the goal is to get the data in a format where it can be retrieved for value later.

Level 1, Character Splitting, represents the starting point, focusing on the physical positioning and structure of the text chunks.

## 2. Character Splitting Concept

**Character splitting is the most basic form of splitting**.

### 2.1 Definition
Character splitting is the method where you **split your documents by a fixed static character length**.

### 2.2 Pros and Cons
| Category | Detail | Source |
| :--- | :--- | :--- |
| **Pros** | It is **extremely simple and easy**. | |
| **Cons** | It is **very rigid**. | |
| **Cons** | It **doesn't take into account the structure of your text**. | |
| **Practical Use** | The speaker noted that they do not know anybody who uses this method in production. | |

### 2.3 Key Concepts

Two key concepts introduced in Level 1 are **Chunk Size** and **Chunk Overlap**.

1.  **Chunk Size:** This defines the static character limit used for the split.
2.  **Chunk Overlap:** This defines the number of characters that the end of one chunk shares with the beginning of the next chunk. For example, if the overlap is 4 characters, the **last four characters of Chunk one will be the same four characters of Chunk two**. Overlap helps retain context across chunk boundaries.

## 3. Implementation and Examples

### 3.1 Manual Implementation Example

To appreciate the nuances of chunking, the speaker demonstrated a manual implementation using the following example text:

`This is the text I would like to Chunk Up it is an example text for this exercise`

If we set the **chunk size to 35 characters**:

*   The manual split results in chunks that often **split in the middle of a word** (e.g., "This is the text I would like to CH Unk up...").
*   The fundamental issue is that using a fixed character length without respecting word boundaries or sentence structure leads to poor results.

### 3.2 LangChain Character Splitter

The LangChain library provides a `CharacterTextSplitter` that performs the same function.

#### Initialization Parameters:
When initializing the splitter, you define:
*   `chunk_size`: How large the chunks should be (e.g., 35 characters).
*   `chunk_overlap`: How many characters should overlap between adjacent chunks (e.g., 4 characters).
*   `separator`: Setting this to an **empty string** (`""`) means the splitter will split **by character**.

#### Output Format:
When using LangChain’s `create_documents` function, the splitter returns a list of **document objects**. A document object holds the string content (in `page_content`) but also includes valuable **metadata**.

#### Separator Effects:
If a separator other than an empty string is specified (e.g., using "CH") and the text contains that separator, the separator itself is **removed** when the text is split. This is generally not advisable unless you know exactly what you are doing.

### 3.3 Visualization Tool (chunkviz.com)

The speaker created a tool called **chunk viz.com** to visually demonstrate different chunking techniques. This tool allows users to input text and adjust the chunk sizes and overlaps to visualize the resulting splits and overlapping sections.

For instance, using the example text with a chunk size of 35 on the tool shows the text ending prematurely, confirming the manual result where the split occurred in the middle of a word. Increasing the overlap setting visually introduces the shared section between chunks.