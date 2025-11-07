import faiss, json
from sentence_transformers import SentenceTransformer
from llmCall import query

index = faiss.read_index("dataIndexed.faiss")

with open("dataChunks.json", "r") as f:
    all_chunks = json.load(f)

model = SentenceTransformer('models/all-MiniLM-L6-v2')

userQuery = "What is a hybrid car?"

query_vector = model.encode([userQuery]).astype('float32')
D, I = index.search(query_vector, k=3)
retrieved_chunks = [all_chunks[idx] for idx in I[0]]
context = "\n\n".join(retrieved_chunks)

prompt = f"""
<SYSTEM>
You are an expert assistant. 
Use ONLY the following context to answer the question. 
Do not rely on any prior knowledge.
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