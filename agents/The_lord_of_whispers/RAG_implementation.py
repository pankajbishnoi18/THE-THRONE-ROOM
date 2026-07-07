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


def retrieve(situation:str,query: str, top_k: int = 5):
    
    query_embedding = model.encode(situation)
    chunks, embedding = get_archive_embeddings()
    l = []
    
    for i in range(len(embedding)):
        score = cosine_similarity(query_embedding, embedding[i])
        s = [float(score), chunks[i]]
        
        l.append(s)
        
    l = sorted(l)[::-1]
    
    
    
    small_l=smaller_chunk_maker(l[:top_k],query_embedding)
    
    final_llm_ready=compress_and_convert_into_one(small_l)
    return final_llm_ready

def smaller_chunk_maker(l,query_embedding):
    small_chunks=[[],[],[],[],[]]
    k=[[],[],[],[],[]]
    metadata=[[],[],[],[],[]]
    for i in range(len(l)):
        small_chunks[i]+=fixed_size_chunk(l[i][1])[0]
        metadata[i]+=fixed_size_chunk(l[i][1])[1]
    
    
        smaller_emb=embeddings(small_chunks[i])

        k[i]=retrieve_for_small(smaller_emb,query_embedding,small_chunks[i],metadata[i])
    return k
    
    
def retrieve_for_small(smaller_emb,query_embedding,small_chunks,metadata_of_file):
    
    l=[]
    for i in range(len(smaller_emb)):
        score = cosine_similarity(query_embedding, smaller_emb[i])
        s = [float(score),metadata_of_file[i], small_chunks[i]]
        if score>0.5:
            l.append(s)
    l = sorted(l)[::-1]
    return l[:4]

        
def fixed_size_chunk(text:str,chunk_size:int=500,overlap:int=400):
    words=text.split()
    
    l=[]
    meta_store=[]
    chunk=0
    start_chunk=0
    while True:
        metadata={"id":None,"start_word":None,"end_word":None} 
        if start_chunk>len(words):
            break
        
        l.append([])
        chunk+=1
        metadata["id"]=chunk
        metadata["start_word"]=start_chunk
       
    
        for i in range(start_chunk,start_chunk+chunk_size):
            if i==len(words):
                metadata["end_word"]=i
                break
            metadata["end_word"]=start_chunk+chunk_size
            l[chunk-1].append(words[i])
           
        start_chunk+=chunk_size-overlap
        meta_store.append(metadata)
        
        
    chunks=[] 
     
    for i in l:
        
        chunk=" ".join(i)
        chunks.append(chunk)
       
    return chunks,meta_store

def chunk_maker():
    chunks=[]
    path1=Path.cwd()/"lore2"/"common"
    path2=Path.cwd()/"lore2"/"lord_of_whispers"
    for file in path1.iterdir():
        chunks.append(read_file(file))
    for file in path2.iterdir():
        chunks.append(read_file(file))
    return chunks
    

def compress_and_convert_into_one(response):
    merged_documents = []
    for file_chunks in response:

        if not file_chunks:
            continue

        file_chunks = sorted(
            file_chunks,
            key=lambda x: x[1]["start_word"]
        )

        merged_words = []
        current_end = 0

        for _, meta, text in file_chunks:

            words = text.split()

            start = meta["start_word"]
            end = meta["end_word"]

            
            overlap = max(0, current_end - start)

            
            merged_words.extend(words[overlap:])

            
            current_end = max(current_end, end)

        merged_documents.append(" ".join(merged_words))

    return merged_documents
