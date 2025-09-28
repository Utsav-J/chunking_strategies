## 1. Context and Goal

Level 4, **Semantic Splitting**, is a conceptual shift away from the "naive ways of splitting" (Levels 1, 2, and 3) which only focused on the physical structure and positioning of text. Semantic splitting attempts to delve into the **actual meaning and context** of the text. This process is analogous to categorizing books by their genre or themes rather than just their physical size.

The method relies on an **embedding-based chunking approach**, which is noted as being more expensive, requiring more work, and definitely slower than the methods in the first three levels.

### Core Hypothesis
The central hypothesis is based on comparing the distance between embeddings derived from segments of the document:

*   If two embeddings are **close** distance-wise, they are assumed to be talking about the **same thing**.
*   If two embeddings are **farther** apart, they are potentially **not talking about the same thing**, indicating a semantic shift. A large distance suggests an ideal **break point** where the chunk should be split.

## 2. Exploration of Implementation Strategies

The speaker explored two primary embedding-based methods:

1.  **Hierarchical Clustering with Positional Reward:** The initial thought was to apply a clustering algorithm to sentence embeddings, treating the resulting clusters as chunks. A "positional reward" was added to ensure short sentences following long ones were included, maintaining context. This method was found to be "messy to work with" and not intuitively tunable, leading to its rejection.

2.  **Finding Break Points Between Sequential Sentences (Distance Comparison):** This was the method chosen for detailed step-by-step implementation and demonstration.

## 3. Step-by-Step Approach: Sequential Distance Method

The following steps detail the preferred method for semantic splitting, as demonstrated using a Paul Graham essay containing 317 sentences.

### Step 1: Sentence Splitting and Structuring

1.  The long document (Paul Graham essay) is first split into individual sentences (e.g., using regex on periods, question marks, or explanation points).
2.  The resulting sentences are structured into a list of dictionaries, where each entry contains the raw sentence text and its index.

### Step 2: Combining Sentences for Grouped Embeddings

1.  To reduce noise and movement in the embedding space, individual sentences are **combined into groups** before embedding.
2.  This grouping process compares the embedding of a combined section (e.g., Sentence 1, 2, and 3) with the embedding of the next sequential section (e.g., Sentence 2, 3, and 4).
3.  A small function is used with a configurable **buffer size** (e.g., one sentence before and one sentence after) to create these `combined sentence` strings.

### Step 3: Generating Embeddings

1.  The text from the `combined sentence` key is passed to an embedding model (OpenAI embeddings were used in the demonstration).
2.  The resulting vector is stored back into the document list under a new key, `combined sentence embeddings`.

### Step 4: Calculating Distance to Next

1.  A crucial metric, **"Distance to Next,"** is calculated.
2.  For each combined sentence embedding, the distance is measured between the **current embedding** and the **next sequential embedding**.
3.  This measures the semantic jump between adjacent groups of text (e.g., how far is Group A from Group B). These distances are collected into a list.

### Step 5: Plotting Distances and Setting Threshold

1.  The calculated distances are plotted sequentially to visually identify semantic shifts. Peaks in the plot indicate outliers—points where the current text group is highly dissimilar from the following text group.
2.  A **breakpoint percentile threshold** is established to programmatically define outliers. In the demonstration, the 95th percentile was used, meaning only the top 5% of the calculated distances were considered significant breaks.
3.  A line is drawn on the plot representing this threshold. Any point above this line is designated as a break point.

### Step 6: Identifying Break Points and Creating Final Chunks

1.  The indices corresponding to the distances that exceed the percentile threshold are collected. These indices represent where the content shifts in meaning.
2.  These indices are used to segment the original sentences, combining all sentences between two break points into a single, cohesive, semantically determined chunk.
3.  For example, if the index 23 is identified as a break point, sentences 0 through 23 are combined to form the first chunk.

This process results in chunks of **different lengths** that group content based on semantic similarity, moving beyond fixed-size or structure-based splitting. The speaker views this as a promising, albeit experimental, direction for future chunking strategies as compute and language models improve.