import os
import sys
current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from langchain.text_splitter import MarkdownTextSplitter
default_filepath = os.path.join(root_dir,"README.md")

def chunk_markdown(md_file:str=default_filepath,chunk_size:int=200, chunk_overlap:int=25):
    markdown_spitter = MarkdownTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    md_text = ""
    with open(md_file,"r", encoding="utf-8") as f:
        md_text=f.read()
    chunks = markdown_spitter.create_documents([md_text])
    print("Original Word Count")
    print(len(md_text))
    print("Number of chunks")
    print(len(chunks))
    print("----------------")
    print(chunks)
    return chunks

if __name__ == '__main__':
    chunk_markdown()