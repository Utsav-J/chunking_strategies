import os
import sys
current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from langchain.text_splitter import PythonCodeTextSplitter
default_filepath = os.path.join(current_dir,"markdown.py")

def chunk_python(python_filepath:str=default_filepath,chunk_size:int=200, chunk_overlap:int=25):
    python_splitter = PythonCodeTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    codebase = ""
    with open(python_filepath,"r", encoding="utf-8") as f:
        codebase=f.read()
    chunks = python_splitter.create_documents([codebase])
    print("Original Word Count")
    print(len(codebase))
    print("Number of chunks")
    print(len(chunks))
    print("----------------")
    print(chunks)
    return chunks

if __name__ == '__main__':
    chunks = chunk_python()
    for doc in chunks:
        print("\n\n",doc.page_content,"\n-----------------\n")