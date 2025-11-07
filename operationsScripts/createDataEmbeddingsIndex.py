import os, json
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import faiss

def chunk_text(text, chunk_size=1024, overlap_ratio=0.1):
    chunks = []
    start = 0
    text_length = len(text)
    overlap = int(chunk_size * overlap_ratio)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # move ahead keeping 10% overlap

    return chunks

all_chunks = []

def dataAddition():
    for filename in os.listdir("data"):
        if filename.endswith(".txt"):
            with open(f"data/{filename}", "r", encoding="utf-8") as f:
                text = f.read()
                chunks = chunk_text(text)
                all_chunks.extend(chunks)

    with open("dataChunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    
    return all_chunks

def createEmbeddings():
    with open("dataChunks.json", "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    model = SentenceTransformer('models/all-MiniLM-L6-v2')
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    data = [{"text": text, "embedding": emb.tolist()} for text, emb in zip(all_chunks, embeddings)]
    with open("dataEmbeddings.pkl", "wb") as f:
        pickle.dump(data, f)
    return embeddings

def createIndex():
    with open("dataEmbeddings.pkl", "rb") as f:
        data = pickle.load(f)
    texts = [d["text"] for d in data]
    embeddings = np.array([d["embedding"] for d in data]).astype("float32")

    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, 256, 8, 4, faiss.METRIC_INNER_PRODUCT)
    # nlist - groups all thw vectors into nlist clusters -> Higher nlist = better accuracy but more memory and slower training.
    # m (pq_m) - Each embedding vector (say 384-D) is split into pq_m smaller parts. Example: 384 dimensions → 8 parts → each subvector = 48-D.
    # pq_bits - Defines how many bits FAISS uses to represent each subvector after quantization. If you set pq_bits = 10 and pq_m = 8, then each vector uses 8 × 10 = 80 bits = 10 bytes of storage.

    # A bit of debugging shows nx==111 (good) and k==256 (1 << nbits in ProductQuantizer::set_derived_values). I can fix the issue by setting the number of bits per subvector to 4, instead of 8, as in:
    index.train(embeddings)
    index.add(embeddings)
    faiss.write_index(index, "dataIndexed.faiss")
