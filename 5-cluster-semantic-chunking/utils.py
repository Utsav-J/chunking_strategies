import tiktoken
from langchain_community.embeddings import HuggingFaceEmbeddings

def token_count(text:str)->int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text,disallowed_special=()))
    return num_tokens

def get_hf_embedding_function():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    def embedding_function(texts:list[str]):
        embedding_model.embed_documents(texts)

    return embedding_function