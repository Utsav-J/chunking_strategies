import numpy as np
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from utils import token_count,get_hf_embedding_function
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self,text:str)->List[str]:
        pass

class ClusterSemanticChunker(BaseChunker):
    def __init__(self, embedding_function=None, max_chunk_size=400, min_chunk_size=50, length_function=token_count):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=length_function,
            separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        self._chunk_size = max_chunk_size
        self.max_cluster_size = max_chunk_size // min_chunk_size
        if embedding_function is None:
            embedding_function = get_hf_embedding_function()
        self.embedding_function = embedding_function
    # ... keep all your existing code unchanged ...

    # ---------------- Helper Functions ---------------- #

    def visualize_similarity_matrix(self, similarity_matrix, sentences=None):
        """
        Display the similarity matrix as a heatmap.
        Optionally annotate with sentence indices or first few words.
        """
        plt.figure(figsize=(8,6))
        sns.heatmap(similarity_matrix, cmap="viridis")
        if sentences:
            labels = [s[:15] + "..." if len(s) > 15 else s for s in sentences]
            plt.xticks(ticks=np.arange(len(labels))+0.5, labels=labels, rotation=90)
            plt.yticks(ticks=np.arange(len(labels))+0.5, labels=labels, rotation=0)
        plt.title("Sentence Similarity Matrix")
        plt.show()

    def print_clusters(self, sentences, clusters):
        """
        Nicely print the sentences grouped into clusters.
        """
        for i, (start, end) in enumerate(clusters):
            chunk = ' '.join(sentences[start:end+1])
            print(f"\n--- Chunk {i+1} ({start}-{end}) ---")
            print(chunk)

    def visualize_reward_matrix(self, similarity_matrix, max_cluster_size):
        """
        Compute and plot the reward matrix for all possible clusters.
        """
        n = similarity_matrix.shape[0]
        reward_matrix = np.zeros((n,n))
        for start in range(n):
            for end in range(start, min(n, start+max_cluster_size)):
                reward_matrix[start,end] = self._calculate_reward(similarity_matrix, start, end)
        
        plt.figure(figsize=(8,6))
        sns.heatmap(reward_matrix, cmap="coolwarm")
        plt.title("Reward Matrix (sum of sub-matrix similarities)")
        plt.xlabel("End Sentence Index")
        plt.ylabel("Start Sentence Index")
        plt.show()

    def _get_similarity_matrix(self, embedding_function, sentences):
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        BATCH_SIZE = 500    
        N = len(sentences)
        embedding_matrix = None

        for i in range(0,N, BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]
            embeddings = embedding_model.embed_documents(batch_sentences)
            batch_embedding_matrix = np.array(embeddings)
            print(embeddings)
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)
        
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T) #type: ignore
        return similarity_matrix

    def _calculate_reward(self, matrix, start, end):
        sub_matrix = matrix[start:end+1, start:end+1]
        return np.sum(sub_matrix)
    
    def _optimal_segmentation(self, matrix, max_cluster_size,window_size=3):
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value 
        np.fill_diagonal(matrix, 0) 

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    # local_density = calculate_local_density(matrix, i, window_size)
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    # Adjust reward based on local density
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size]
                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters
    
    def split_text(self, text: str) -> List[str]:
        sentences = self.splitter.split_text(text)
        similarity_matrix = self._get_similarity_matrix(self.embedding_function, sentences)
        clusters = self._optimal_segmentation(similarity_matrix, self.max_cluster_size)
        docs = [' '.join(sentences[start:end+1]) for start, end in clusters]
        self.visualize_similarity_matrix(similarity_matrix, sentences)
        self.visualize_reward_matrix(similarity_matrix, self.max_cluster_size)
        self.print_clusters(sentences, clusters)
        return docs
        