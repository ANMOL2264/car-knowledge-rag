import faiss, json, os
from llmCall import query
from embeddingsCall import getEmbeddings
import numpy as np

def retrieve_top_k(index, q_vec, k=40, similarity_threshold=0.20):
    index.nprobe = 8
    print(q_vec.shape)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k) 
    sims = D[0]   # array of similarities or distances depending on index setup
    idxs = I[0]
    # Keep only those above threshold
    candidates = []
    for sim, idx in zip(sims, idxs):
        if sim >= similarity_threshold:
            candidates.append((idx, sim))
    return candidates  # list of (index, similarity)

index = faiss.read_index("car-knowledge-rag\dataIndexed.faiss")
print("Got Index")
with open("car-knowledge-rag\dataChunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)
print("Got Data")

print(len(all_chunks))

userQuery = "why brakes were added in cars ? Explain in detail."

# query_vector = model.encode([userQuery]).astype('float32')
query_vector = getEmbeddings(userQuery)
retrivedChunksList = retrieve_top_k(index, np.array(query_vector, dtype=np.float32).reshape(1, -1), 20, 0.55)
retrivedChunks = [(all_chunks[k[0]], k[1]) for k in retrivedChunksList]
textPart = [k[0] for k in retrivedChunks]
print(len(textPart))
context = "\n\n".join(textPart)

with open(r"car-knowledge-rag\test\contextTest.txt", "w", encoding="utf-8") as f:
    f.write(context)

prompt = f"""
<SYSTEM>
You are an expert assistant. 
Use the provided context strictly to answer the question.
Use only the information provided in the context to answer, but do not mention the context or that you used it. Write your answer as if you already knew the information. 
Provide only the final factual answer in natural language.
If the answer is not in the context, say "I don't have that information."
</SYSTEM>
<CONTEXT>
{context}
</CONTEXT>
<USER>
{userQuery}
</USER>
"""

response, responseMetaData = query(prompt)

print(response)

with open(r"car-knowledge-rag\test\responseTest.txt", "w", encoding="utf-8") as f:
    f.write(response)