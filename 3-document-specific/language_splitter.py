import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
default_filepath = os.path.join(current_dir, "test.js")


def chunk_it(
    js_filepath: str = default_filepath,
    chunk_size: int = 100,
    chunk_overlap: int = 25,
):
    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    codebase = ""
    with open(js_filepath, "r", encoding="utf-8") as f:
        codebase = f.read()
    chunks = js_splitter.create_documents([codebase])
    print("Original Word Count")
    print(len(codebase))
    print("Number of chunks")
    print(len(chunks))
    print("----------------")
    print(chunks)
    return chunks


if __name__ == '__main__':
    chunks = chunk_it()
    for doc in chunks:
        print("\n\n",doc.page_content,"\n-----------------\n")