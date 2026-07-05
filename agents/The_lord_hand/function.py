from pathlib import Path
from sentence_transformers import SentenceTransformer 
import numpy as np


def read_file(path):
    return Path(path).read_text(encoding="utf-8")



model = SentenceTransformer("BAAI/bge-small-en-v1.5")
def embeddings(chunks):
    embedding=model.encode(chunks)
    return embedding

def cosine_similarity(a, b):
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return float(np.dot(a, b) / (mag_a * mag_b))


    
_chunks_cache=None
_embeddings_cache=None


def get_archive_embeddings():
    global _chunks_cache, _embeddings_cache
    if _chunks_cache is None:
        _chunks_cache = chunk_maker()
        _embeddings_cache = embeddings(_chunks_cache)
    return _chunks_cache, _embeddings_cache


def retrieve(query: str, top_k: int = 4):
    query_embedding = model.encode(query)
    chunks, embedding = get_archive_embeddings()
    l = []
    
    for i in range(len(embedding)):
        score = cosine_similarity(query_embedding, embedding[i])
        s = [float(score), chunks[i]]
        if score>0.65:
            l.append(s)
        
    l = sorted(l)[::-1]
    
    return l[:top_k]

def chunk_maker():
    chunks=[]
    path1=Path.cwd()/"lore2"/"common"
    path2=Path.cwd()/"lore2"/"lord_hand"
    for file in path1.iterdir():
        chunks.append(read_file(file))
    for file in path2.iterdir():
        chunks.append(read_file(file))
    return chunks
    

