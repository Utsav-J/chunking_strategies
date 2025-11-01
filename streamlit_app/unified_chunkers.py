"""
Unified chunking interface for all chunking strategies in the project.
This module wraps all chunking implementations into a consistent interface.
"""
import os
import sys
from typing import List, Dict, Any
import re

# Add parent directory to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import chunking modules
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)

# For semantic and cluster chunking
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAVE_EMBEDDINGS = True
except ImportError:
    HAVE_EMBEDDINGS = False

try:
    import tiktoken
    HAVE_TIKTOKEN = True
except ImportError:
    HAVE_TIKTOKEN = False

HAVE_SEMANTIC = HAVE_NUMPY and HAVE_EMBEDDINGS and HAVE_SKLEARN
HAVE_CLUSTER = HAVE_NUMPY and HAVE_EMBEDDINGS and HAVE_TIKTOKEN


class ChunkingResult:
    """Container for chunking results"""
    def __init__(self, chunks: List[str], metadata: Dict[str, Any] = None):
        self.chunks = chunks
        self.metadata = metadata or {}
        self.num_chunks = len(chunks)
        self.avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0


class UnifiedChunker:
    """Unified interface for all chunking strategies"""
    
    def __init__(self):
        # Initialize embedding models if available
        if HAVE_SEMANTIC:
            self._init_semantic_embeddings()
        if HAVE_CLUSTER:
            self._init_cluster_embeddings()
    
    def _init_semantic_embeddings(self):
        """Initialize embedding model for semantic chunking"""
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    
    def _init_cluster_embeddings(self):
        """Initialize embedding model for cluster semantic chunking"""
        self.cluster_embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
    
    # ============ Strategy 1: Character Chunking ============
    
    def character_chunking(self, text: str, chunk_size: int = 100, chunk_overlap: int = 0) -> ChunkingResult:
        """
        Fixed-size character chunking.
        Simple, fast, but may cut sentences mid-way.
        """
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.create_documents([text])
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        return ChunkingResult(
            chunks=chunk_texts,
            metadata={
                "strategy": "Character Chunking",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "description": "Fixed-size character splitting"
            }
        )
    
    # ============ Strategy 2: Recursive Character Chunking ============
    
    def recursive_character_chunking(self, text: str, chunk_size: int = 200, chunk_overlap: int = 20) -> ChunkingResult:
        """
        Recursive character text splitter.
        Splits by multiple separators (paragraphs, sentences, etc.) in order.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.create_documents([text])
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        return ChunkingResult(
            chunks=chunk_texts,
            metadata={
                "strategy": "Recursive Character Chunking",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "description": "Hybrid approach with multiple separators"
            }
        )
    
    # ============ Strategy 3: Document-Specific Chunking ============
    
    def markdown_chunking(self, text: str, chunk_size: int = 200, chunk_overlap: int = 25) -> ChunkingResult:
        """Markdown-specific chunking"""
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.create_documents([text])
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        return ChunkingResult(
            chunks=chunk_texts,
            metadata={
                "strategy": "Markdown Chunking",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "description": "Markdown-aware splitting"
            }
        )
    
    def python_chunking(self, text: str, chunk_size: int = 200, chunk_overlap: int = 25) -> ChunkingResult:
        """Python code-specific chunking"""
        splitter = PythonCodeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.create_documents([text])
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        return ChunkingResult(
            chunks=chunk_texts,
            metadata={
                "strategy": "Python Chunking",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "description": "Python code-aware splitting"
            }
        )
    
    # ============ Strategy 4: Semantic Chunking ============
    
    def semantic_chunking(self, text: str, buffer_size: int = 1, breakpoint_percentile: float = 95.0) -> ChunkingResult:
        """
        Semantic chunking using embedding-based distance analysis.
        Finds natural breakpoints based on semantic similarity.
        """
        if not HAVE_SEMANTIC:
            raise ImportError("Semantic chunking requires numpy, sklearn, and HuggingFace embeddings")
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Combine sentences with buffer
        combined_sentences = self._combine_sentences(sentences, buffer_size)
        
        # Embed sentences
        embeddings = self.embedding_model.embed_documents(
            [s['combined_sentence'] for s in combined_sentences]
        )
        for i, sentence in enumerate(combined_sentences):
            sentence['embedding'] = embeddings[i]
        
        # Calculate distances
        distances, sentences_with_dist = self._calculate_cosine_distances(combined_sentences)
        
        # Find breakpoints
        if not HAVE_NUMPY:
            raise ImportError("NumPy is required for semantic chunking")
        breakpoint_threshold = np.percentile(distances, breakpoint_percentile)
        breakpoints = [i for i, d in enumerate(distances) if d > breakpoint_threshold]
        
        # Create chunks
        chunks = self._combine_into_chunks(sentences_with_dist, breakpoints)
        
        return ChunkingResult(
            chunks=chunks,
            metadata={
                "strategy": "Semantic Chunking",
                "buffer_size": buffer_size,
                "breakpoint_percentile": breakpoint_percentile,
                "num_sentences": len(sentences),
                "breakpoints": len(breakpoints),
                "description": "Embedding-based semantic splitting"
            }
        )
    
    def _split_sentences(self, text: str) -> List[Dict]:
        """Split text into sentences"""
        sentences_list = re.split(r'(?<=[.?!])\s+', text)
        return [{'sentence': s, 'index': i} for i, s in enumerate(sentences_list)]
    
    def _combine_sentences(self, sentences: List[Dict], buffer_size: int = 1) -> List[Dict]:
        """Combine sentences with buffer"""
        for i, item in enumerate(sentences):
            start = max(0, i - buffer_size)
            end = min(len(sentences), i + buffer_size + 1)
            combined = " ".join(s["sentence"] for s in sentences[start:end])
            item["combined_sentence"] = combined
        return sentences
    
    def _calculate_cosine_distances(self, sentences: List[Dict]) -> tuple:
        """Calculate cosine distances between consecutive sentences"""
        distances = []
        for i in range(len(sentences) - 1):
            emb_current = sentences[i]['embedding']
            emb_next = sentences[i + 1]['embedding']
            similarity = cosine_similarity([emb_current], [emb_next])[0][0]
            distance = 1 - similarity
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance
        return distances, sentences
    
    def _combine_into_chunks(self, sentences: List[Dict], breakpoints: List[int]) -> List[str]:
        """Combine sentences into chunks based on breakpoints"""
        chunks = []
        start_idx = 0
        
        for breakpoint in breakpoints:
            end_idx = breakpoint
            chunk = ' '.join(s['sentence'] for s in sentences[start_idx:end_idx + 1])
            chunks.append(chunk)
            start_idx = breakpoint + 1
        
        # Add remaining sentences
        if start_idx < len(sentences):
            chunk = ' '.join(s['sentence'] for s in sentences[start_idx:])
            chunks.append(chunk)
        
        return chunks
    
    # ============ Strategy 5: Cluster Semantic Chunking ============
    
    def cluster_semantic_chunking(
        self, 
        text: str, 
        max_chunk_size: int = 400,
        min_chunk_size: int = 50
    ) -> ChunkingResult:
        """
        Cluster-based semantic chunking using global optimization.
        More sophisticated than regular semantic chunking.
        """
        if not HAVE_CLUSTER:
            raise ImportError("Cluster semantic chunking requires additional dependencies")
        
        # Initial split with recursive character splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        initial_chunks = splitter.split_text(text)
        
        # Get embeddings
        embeddings = self.cluster_embedding_model.embed_documents(initial_chunks)
        
        # Compute similarity matrix
        if not HAVE_NUMPY:
            raise ImportError("NumPy is required for cluster semantic chunking")
        similarity_matrix = np.dot(np.array(embeddings), np.array(embeddings).T)
        
        # Find optimal clusters (simplified version)
        max_cluster_size = max_chunk_size // min_chunk_size
        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size)
        
        # Combine into final chunks
        chunks = [' '.join(initial_chunks[start:end+1]) for start, end in clusters]
        
        return ChunkingResult(
            chunks=chunks,
            metadata={
                "strategy": "Cluster Semantic Chunking",
                "max_chunk_size": max_chunk_size,
                "min_chunk_size": min_chunk_size,
                "num_initial_chunks": len(initial_chunks),
                "description": "Global optimization for semantic coherence"
            }
        )
    
    def _optimal_segmentation(self, matrix, max_cluster_size: int) -> List[tuple]:
        """Find optimal segmentation using dynamic programming"""
        if not HAVE_NUMPY:
            raise ImportError("NumPy is required for cluster semantic chunking")
            
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value
        np.fill_diagonal(matrix, 0)
        
        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)
        
        for i in range(n):
            for size in range(1, min(max_cluster_size + 1, i + 2)):
                if i - size + 1 >= 0:
                    reward = np.sum(matrix[i - size + 1:i + 1, i - size + 1:i + 1])
                    if i - size >= 0:
                        reward += dp[i - size]
                    if reward > dp[i]:
                        dp[i] = reward
                        segmentation[i] = i - size + 1
        
        # Reconstruct clusters
        clusters = []
        i = n - 1
        while i >= 0:
            start = int(segmentation[i])
            clusters.append((start, i))
            i = start - 1
        
        clusters.reverse()
        return clusters


def get_available_strategies() -> Dict[str, Any]:
    """Get all available chunking strategies with their configurations"""
    strategies = {
        "Character Chunking": {
            "description": "Fixed-size character splitting",
            "params": ["chunk_size", "chunk_overlap"],
            "defaults": {"chunk_size": 100, "chunk_overlap": 0},
            "available": True
        },
        "Recursive Character Chunking": {
            "description": "Hybrid approach with multiple separators",
            "params": ["chunk_size", "chunk_overlap"],
            "defaults": {"chunk_size": 200, "chunk_overlap": 20},
            "available": True
        },
        "Markdown Chunking": {
            "description": "Markdown-aware splitting",
            "params": ["chunk_size", "chunk_overlap"],
            "defaults": {"chunk_size": 200, "chunk_overlap": 25},
            "available": True
        },
        "Python Chunking": {
            "description": "Python code-aware splitting",
            "params": ["chunk_size", "chunk_overlap"],
            "defaults": {"chunk_size": 200, "chunk_overlap": 25},
            "available": True
        },
        "Semantic Chunking": {
            "description": "Embedding-based semantic splitting",
            "params": ["buffer_size", "breakpoint_percentile"],
            "defaults": {"buffer_size": 1, "breakpoint_percentile": 95.0},
            "available": HAVE_SEMANTIC
        },
        "Cluster Semantic Chunking": {
            "description": "Global optimization for semantic coherence",
            "params": ["max_chunk_size", "min_chunk_size"],
            "defaults": {"max_chunk_size": 400, "min_chunk_size": 50},
            "available": HAVE_CLUSTER
        }
    }
    return strategies


if __name__ == "__main__":
    # Test the chunkers
    test_text = """
    One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.
    
    Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.
    
    It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer.
    """
    
    chunker = UnifiedChunker()
    
    # Test character chunking
    result = chunker.character_chunking(test_text, chunk_size=100, chunk_overlap=0)
    print(f"Character Chunking: {result.num_chunks} chunks")
    print(f"Avg size: {result.avg_chunk_size:.0f} chars")
    
    # Test recursive chunking
    result = chunker.recursive_character_chunking(test_text, chunk_size=200, chunk_overlap=20)
    print(f"\nRecursive Chunking: {result.num_chunks} chunks")
    print(f"Avg size: {result.avg_chunk_size:.0f} chars")
    
    # Print available strategies
    print("\nAvailable strategies:")
    strategies = get_available_strategies()
    for name, info in strategies.items():
        status = "✓" if info["available"] else "✗"
        print(f"  {status} {name}: {info['description']}")

