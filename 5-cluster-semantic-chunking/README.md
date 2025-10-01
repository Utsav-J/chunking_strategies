![LOGIC](https://raw.githubusercontent.com/ALucek/chunking-strategies/9e5730aadc6ee71e9798a4c9df0384c9743969b5/media/cluster_chunking.png)

### ClusterSemanticChunker: A Global Approach to Text Chunking

The **ClusterSemanticChunker** adopts a **global optimization strategy** for chunking, standing in contrast to Kamradt’s method, which makes **local decisions** about split points. Instead of analyzing text through a sliding window of context, it considers **relationships between all text pieces simultaneously**, aiming to create the most **semantically coherent groupings** while respecting size constraints.

#### 1. Initial Splitting

The process starts similarly to other chunkers: the text is divided into **small fixed-size pieces** (default ≈ 50 tokens) using **recursive splitting**. However, unlike local methods, the ClusterSemanticChunker does not only focus on consecutive pieces.

#### 2. Constructing the Similarity Matrix

Each piece is embedded into a vector space, and **cosine similarities** are calculated between all possible pairs. This produces a **similarity matrix** that provides a **complete view of semantic relationships** across the entire document.

#### 3. Dynamic Programming for Optimal Chunking

Using the similarity matrix, the chunker employs **dynamic programming** to determine the **optimal grouping** of pieces into chunks:

* For each position in the text, the algorithm considers **different chunk sizes**.
* A **reward function** evaluates the total semantic similarity of all pieces within a potential chunk.
* Intermediate results are stored to efficiently explore all possible chunkings and identify the **global optimum**.

#### 4. Enforcing Size Constraints

Chunks are restricted by a **maximum number of pieces** (`max_cluster`). Within this limit, the algorithm **maximizes semantic coherence**, producing more **natural groupings** than approaches that rely solely on local context. This allows it to identify relationships between pieces that are **far apart** in the text but **closely related semantically**.

#### 5. Advantages Over Sliding Window Approaches

By taking a global view, the ClusterSemanticChunker:

* Avoids the pitfalls of local methods that may miss **related content separated by brief topic shifts**.
* Produces chunks that **better reflect the document’s semantic structure**.
* Maintains **practical size limits** suitable for downstream processing.
