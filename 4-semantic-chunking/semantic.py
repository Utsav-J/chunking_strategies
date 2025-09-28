import re
import os
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

current_dir = os.path.dirname(__file__)
default_txt_file_path = os.path.join(current_dir,"mit.txt")

def split_sentences(text:str):
    # splitting the essay on '.', '?', and '!'
    single_sentences_list = re.split(r'(?<=[.?!])\s+', text)
    print (f"{len(single_sentences_list)} sentences were found\n")
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]
    # [
    #     {"sentence": "\n\nWant to start a startup?", "index": 0},
    #     {"sentence": "Get funded by\nY Combinator.", "index": 1},
    # ]
    return sentences

def combine_sentences(sentences, buffer_size=1):
    for i, item in enumerate(sentences):
        # slice indices with buffer on both sides
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)

        # join all sentence strings in the window
        combined = " ".join(s["sentence"] for s in sentences[start:end])

        # store result
        item["combined_sentence"] = combined

    return sentences

def embed_text(text_parts:list[dict]):
    embeddings = embedding_model.embed_documents([x['combined_sentence'] for x in text_parts])
    for i, sentence in enumerate(text_parts):
        sentence['combined_sentence_embedding'] = embeddings[i]
    return text_parts

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0] #type:ignore

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


def visualize_semantic_chunk_similarities(distances: list):
    plt.plot(distances)

    y_upper_bound = 0.2
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    # We need to get the distance threshold that we'll consider an outlier
    # We'll use numpy .percentile() for this
    breakpoint_percentile_threshold = 95
    breakpoint_distance_threshold = np.percentile(
        distances, breakpoint_percentile_threshold
    )  # If you want more chunks, lower the percentile cutoff
    plt.axhline(y=float(breakpoint_distance_threshold), color="r", linestyle="-") 

    # Then we'll see how many distances are actually above this one
    num_distances_above_theshold = len(
        [x for x in distances if x > breakpoint_distance_threshold]
    )  # The amount of distances above your threshold
    plt.text(
        x=(len(distances) * 0.01),
        y=y_upper_bound / 50,
        s=f"{num_distances_above_theshold + 1} Chunks",
    )

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_threshold = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]  # The indices of those breakpoints on your list

    # Start of the shading and text
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i, breakpoint_index in enumerate(indices_above_threshold):
        start_index = 0 if i == 0 else indices_above_threshold[i - 1]
        end_index = (
            breakpoint_index if i < len(indices_above_threshold) - 1 else len(distances)
        )

        plt.axvspan(
            start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25
        )
        plt.text(
            x=float(np.average([start_index, end_index])),
            y=float(breakpoint_distance_threshold + (y_upper_bound) / 20),
            s=f"Chunk #{i}",
            horizontalalignment="center",
            rotation="vertical",
        )

    # # Additional step to shade from the last breakpoint to the end of the dataset
    if indices_above_threshold:
        last_breakpoint = indices_above_threshold[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(
                last_breakpoint,
                len(distances),
                facecolor=colors[len(indices_above_threshold) % len(colors)],
                alpha=0.25,
            )
            plt.text(
                x=float(np.average([last_breakpoint, len(distances)])),
                y=float(breakpoint_distance_threshold + (y_upper_bound) / 20),
                s=f"Chunk #{i+1}",
                rotation="vertical",
            )

    plt.title("PG Essay Chunks Based On Embedding Breakpoints")
    plt.xlabel("Index of sentences in essay (Sentence Position)")
    plt.ylabel("Cosine distance between sequential sentences")
    plt.show()
    return indices_above_threshold

def combine_sentences_into_chunks(sentences,indices_above_threshold):
    start_index = 0

    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_threshold:
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        
        # Update the start index for the next group
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)
    
    print("Chunked successfully. Here are some samples:\n\n")
    for i, chunk in enumerate(chunks[:2]):
        buffer = 200
        
        print (f"Chunk #{i}")
        print (chunk[:buffer].strip())
        print ("...")
        print (chunk[-buffer:].strip())
        print ("\n")
    return chunks

def semantic_chunking(filepath:str=default_txt_file_path):
    with open(filepath) as file:
        fulltext = file.read()
    sentences_with_indices  = split_sentences(fulltext)
    combined_sentences = combine_sentences(sentences_with_indices)
    combined_sentences = embed_text(combined_sentences)
    distances, sentences = calculate_cosine_distances(combined_sentences)
    indices_above_threshold = visualize_semantic_chunk_similarities(distances)
    chunks = combine_sentences_into_chunks(sentences,indices_above_threshold)
    print(chunks)
    return chunks

if __name__ == "__main__":
    chunks = semantic_chunking()
    print(f"{len(chunks)} chunks were formed")
    for i in range(len(chunks)):
        print(f"Chunk #{i} : {len(chunks[i])}")