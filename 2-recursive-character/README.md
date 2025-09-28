## Overview

Level 2 introduces **Recursive Character Text Splitting**, a method that moves beyond the naive character limits of Level 1 by starting to look at the **physical structure** of the text. This approach infers what type of chunk sizes are appropriate instead of relying solely on a static character count.

If you are starting a project, the speaker recommends **Level 2 as the go-to splitter** because the return on investment (ROI) for the energy required to split documents is excellent. It is implemented easily using a one-liner, executes quickly, and requires no extra processing.

## Concept: Recursive Character Splitting

Recursive character text splitting works by having a **series of separators** that it uses to recursively go through a document.

### How it Works

The process starts with a long document and iterates through a list of separators to chunk the text.

1.  It begins with the **first separator** (e.g., a double new line) and chunks the entire document based on that separator.
2.  If any resulting chunks are **still too large** (e.g., exceeding the defined chunk size), the process moves to the **next separator** in the list (e.g., a single new line) and recursively attempts to split the remaining large chunks.
3.  The process continues through the list of separators, typically moving from large structural breaks (like paragraphs) down to smaller ones (like spaces) and finally down to individual characters.

The typical series of separators used are:
*   Double new lines
*   New lines
*   Spaces
*   Characters

This means you do not need to specify a fixed character length (like 35 or 200 characters). Instead, you pass the text to the splitter, and it infers the structure.

### The Advantage (Leveraging Human Writing)

This method takes advantage of how humans naturally write text. Since ideas are typically separated by paragraphs, which are often delineated by double new lines, splitting on these structure points helps ensure that:

*   **Paragraphs will hold semantically similar information** that should be kept together.
*   The resulting chunks will be of **different lengths** based on the paragraph size, unlike the fixed-length chunks produced by Level 1 splitting.

The hypothesis behind using this method is that grouping these paragraphs together maintains the contextually and semantically similar information.

## Examples and Implementation

### LangChain Implementation

LangChain provides the `RecursiveCharacterTextSplitter`.

When this splitter uses separators like spaces, the resulting chunks tend to end on **words** more often, meaning the text is generally not splitting in the middle of words like it did in Level 1.

However, even with this method, you may still split in between sentences, which the speaker notes is "not so good". This can be mitigated by increasing the `chunk_size`, which makes it more likely to respect paragraph breaks.

### Example Demonstration

Using example text, if a **chunk size of 450** is used with zero overlap, the splitter successfully groups entire paragraphs together.

The speaker notes that when comparing the length of these resulting paragraphs, they are all of different lengths, demonstrating the key benefit: this method avoids cutting in the middle of content, unlike the naive Level 1 character split, which would cut in the middle of these paragraphs (e.g., a 35-character split).

### Chunk Visualization (`chunkviz.com`)

The speaker demonstrated the recursive splitter using the **chunk viz.com** tool.

*   If the original text is run through a naive character splitter (Level 1) with a size of 35, it results in 26 chunks and splits "all over the place".
*   When selecting the **recursive character text splitter**, adjusting the chunk size reveals a key function: as the chunk size increases, the split will **snap to the nearest word** because the splitter is looking for a space to split on.
*   Increasing the size further shows that the text starts to group multiple paragraphs together, which is possible because the paragraph breaks (double new lines) are one of the specified separators.