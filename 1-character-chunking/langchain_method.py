from langchain.text_splitter import CharacterTextSplitter

def manual(chunk_size:int = 35):
    text = "This is the text I would like to chunk up. It is the example text for this exercise"
    chunks = []
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    print(chunks)
    return chunks

def langchain_method(chunk_size:int=35,chunk_overlap:int=5):
    text = "This is the text I would like to chunk up. It is the example text for this exercise"
    text_splitter = CharacterTextSplitter(separator=" ",chunk_size=chunk_size,chunk_overlap=chunk_overlap,strip_whitespace=False)
    chunks = text_splitter.create_documents([text])
    print(chunks)
    return chunks

if __name__ == "__main__":
    chunks = langchain_method()
