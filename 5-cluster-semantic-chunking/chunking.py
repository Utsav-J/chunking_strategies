import os
from chunkers import ClusterSemanticChunker
from utils import get_hf_embedding_function
chunker = ClusterSemanticChunker(embedding_function=get_hf_embedding_function())

current_dir = os.path.dirname(__file__)
default_txt_file_path = os.path.join(current_dir,"mit.txt")

with open(default_txt_file_path) as file:
    fulltext = file.read()

chunks = chunker.split_text(text=fulltext)