import os
import glob
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

MODEL_DIR = r"D:\Ai\Eduscan\models\all-MiniLM-L6-v2"
if not os.path.exists(MODEL_DIR):
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=MODEL_DIR
    )

EMBEDDING_MODEL = SentenceTransformer(MODEL_DIR)
INDEX_DIR = 'rag_index'

def build_or_load_index(folder='output', persist_dir=INDEX_DIR):
    print("Starting to build or load RAG index...")
    docs = []
    for file in glob.glob(os.path.join(folder, '*.md')):
        with open(file, 'r', encoding='utf-8') as f:
            docs.append({'id': file, 'text': f.read()})

    client = Client(Settings(
        persist_directory=persist_dir,
        anonymized_telemetry=False
    ))
    collection = client.get_or_create_collection('edu_docs')

    if collection.count() == 0 and docs:
        embeddings = [EMBEDDING_MODEL.encode(d['text']) for d in docs]
        collection.add(
            documents=[d['text'] for d in docs],
            metadatas=[{'source': d['id']} for d in docs],
            ids=[d['id'] for d in docs],
            embeddings=embeddings
        )
        print(f"Indexed {len(docs)} documents to '{persist_dir}'")
    else:
        print(f"Loaded existing index with {collection.count()} documents.")
    print("RAG index operation completed.")
    return collection

def retrieve_passages(query, collection, top_k=7):
    print(f"Retrieving top {top_k} passages for query: '{query}'...")
    try:
        q_emb = EMBEDDING_MODEL.encode(query)
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        print("Passage retrieval completed.")
        flat_docs = []
        for doc_group in results['documents']:
            if isinstance(doc_group, list):
                flat_docs.extend(doc_group)
            else:
                flat_docs.append(doc_group)
        return flat_docs
    except Exception as e:
        print(f"Error in passage retrieval: {e}")
        return []
