# backend/ingestion/ingest.py
import os
from app.rag.index import build_or_load_index

if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    build_or_load_index(rebuild=True)
    print("Rebuilt RAG index.")
