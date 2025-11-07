import faiss, json
from sentence_transformers import SentenceTransformer
from llmCall import query

def retrieve_top_k(index, q_vec, k=40, similarity_threshold=0.20):
    index.nprobe = 8
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

index = faiss.read_index("dataIndexed.faiss")
print("Got Index")
with open("dataChunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)
print("Got Data")

print(len(all_chunks))

model = SentenceTransformer('models/all-MiniLM-L6-v2')
print("Got Model")

userQuery = "Tell me something interesting about Suspension in cars ? Explain in detail."

query_vector = model.encode([userQuery]).astype('float32')
retrivedChunksList = retrieve_top_k(index, query_vector, 20, 0.25)
retrivedChunks = [(all_chunks[k[0]], k[1]) for k in retrivedChunksList]
textPart = [k[0] for k in retrivedChunks]
print(len(textPart))
context = "\n\n".join(textPart)

with open("contextTest.txt", "w", encoding="utf-8") as f:
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

with open("responseTest.txt", "w", encoding="utf-8") as f:
    f.write(response)