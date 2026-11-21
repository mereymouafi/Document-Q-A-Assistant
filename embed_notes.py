from math import hypot
from pathlib import Path

from dotenv import load_dotenv
from google import genai

import json


load_dotenv(Path(__file__).with_name(".env"))
client = genai.Client()


def embed(text):
    """Send text to Gemini and get its embedding vector."""
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
    )
    return result.embeddings[0].values
def prepare_document(text):
    """Split the uploaded text into paragraphs and embed each one."""
    text = text.replace("\r\n", "\n")
    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    vectors = [embed(chunk) for chunk in chunks]
    return chunks, vectors


def cosine_similarity(first_vector, second_vector):
    """Compare the direction of two vectors."""
    dot_product = sum(
        first * second
        for first, second in zip(first_vector, second_vector)
    )

    first_length = hypot(*first_vector)
    second_length = hypot(*second_vector)

    return dot_product / (first_length * second_length)




# 1. Read the notes and split them into paragraphs.
notes_path = Path(__file__).with_name("notes.txt")
notes = notes_path.read_text(encoding="utf-8")
chunks = notes.strip().split("\n\n")


index_path = Path(__file__).with_name("notes_index.json")
index = {}

if index_path.exists():
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("Cache vide ou invalide. Reconstruction des embeddings.")
        index = {}

if index.get("chunks") == chunks:
    chunk_vectors = index["vectors"]
    print("Embeddings des notes chargés depuis le fichier.")
else:
    chunk_vectors = []
    for chunk in chunks:
        chunk_vectors.append(embed(chunk))

    index = {"chunks": chunks, "vectors": chunk_vectors}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    print("Embeddings des notes calculés et sauvegardés.")


def ask(question, chunks, chunk_vectors, source_name):
    """Find relevant passages and answer the question with Gemini."""
    question_vector = embed(question)

    scores = []
    for number, chunk_vector in enumerate(chunk_vectors, start=1):
        score = cosine_similarity(question_vector, chunk_vector)
        scores.append(score)
        print(f"Chunk {number}: {score:.3f}")

    top_k = 2
    ranked_indices = sorted(
        range(len(scores)),
        key=scores.__getitem__,
        reverse=True,
    )
    top_indices = ranked_indices[:top_k]

    passages = []
    print("\nPassages retrieved:")

    for citation_number, index in enumerate(top_indices, start=1):
        passage = chunks[index]
        passages.append(f"[{citation_number}]\n{passage}")

        print(
            f"[{citation_number}] {source_name}#chunk-{index + 1} "
            f"(score {scores[index]:.3f})"
        )

    context = "\n\n".join(passages)

    prompt = f"""
Answer the question directly and precisely using only the passages below.
Cite each supporting passage using its numbered label, such as [1] or [2].
Use only the labels shown in the provided passages.
If the passages only support part of the answer, say which part is missing.
If they provide no answer, reply exactly:
I couldn't find the answer in the provided passages.

Question: {question}

Passages:
{context}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )
    return response.text, passages

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer, passages = ask(question, chunks, chunk_vectors, "notes.txt")
        print("\nAnswer:")
        print(answer)