import os
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import BaseNode

def run_split(
        txt_file:str="mit.txt",
        chunk_size:int=200,
        chunk_overlap:int=15
)->list[BaseNode]:
    current_dir = os.path.dirname(__file__)
    txt_file_path = os.path.join(current_dir,txt_file)

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = SimpleDirectoryReader(
        input_files=[txt_file_path]
    ).load_data()

    nodes = splitter.get_nodes_from_documents(documents)
    print(nodes[0])
    return nodes

