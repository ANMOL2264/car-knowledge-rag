import faiss, json
from sentence_transformers import SentenceTransformer
from utils.llmCall import query
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',          # file where logs are saved
    level=logging.INFO,          # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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

def getAnswers(userQuery:str, model, all_chunks, index):

    query_vector = model.encode([userQuery]).astype('float32')
    retrivedChunksList = retrieve_top_k(index, query_vector, 20, 0.25)
    retrivedChunks = [(all_chunks[k[0]], k[1]) for k in retrivedChunksList]
    textPart = [k[0] for k in retrivedChunks]
    context = "\n\n".join(textPart)

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
    return response

index = faiss.read_index("working/utils/dataIndexed.faiss")
model = SentenceTransformer('models/all-MiniLM-L6-v2')
with open("working/utils/dataChunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

while range(5):
    queryText = None
    queryText = input("Enter Your Query : ")
    print(getAnswers(queryText, model, all_chunks, index))