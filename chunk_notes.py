from pathlib import Path

text = Path("notes.txt").read_text(encoding="utf-8")
chunks = text.strip().split("\n\n")

# for number, chunk in enumerate(chunks, start=1):
#     print(f"notes.txt#chunk-{number}")
#     print(chunk)
#     print()

# keyword = input("Enter a search word: ").strip().lower()

# for number, chunk in enumerate(chunks, start=1):
#     if keyword in chunk.lower():
#         print(f"notes.txt#chunk-{number}")
#         print(chunk)


import re

question = input("Ask a question about your AWS notes: ")
question_words = set(re.findall(r"\w+", question.lower()))

best_score = 0
best_number = None
best_chunk = None

for number, chunk in enumerate(chunks, start=1):
    chunk_words = set(re.findall(r"\w+", chunk.lower()))
    matching_words = question_words & chunk_words
    score = len(matching_words)

    if score > best_score:
        best_score = score
        best_number = number
        best_chunk = chunk

if best_chunk is None:
    print("No matching paragraph found.")
else:
    print(f"notes.txt#chunk-{best_number} (score: {best_score})")
    print(best_chunk)