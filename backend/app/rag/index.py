# backend/app/rag/index.py
import os, json, hashlib
from pathlib import Path
from typing import List, Tuple

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parents[3] / '.env'
load_dotenv(dotenv_path)

# Silence Chroma telemetry
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMADB_DISABLE_TELEMETRY", "1")



HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parents[2]                     # .../backend
SEEDS_DIR = BACKEND_DIR / "ingestion" / "seeds"
PERSIST_DIR = BACKEND_DIR / "chroma"
META_PATH = PERSIST_DIR / ".build_meta.json"

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# add near the top:
from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,   # <- silences PostHog errors for good
    allow_reset=True
)

# singletons
_DB: Chroma | None = None
_EMB: OpenAIEmbeddings | None = None

def build_or_load_index(rebuild: bool = False) -> Chroma:
    """
    Build once, then reuse the same Chroma instance (and embeddings) in-process.
    Rebuild only if seeds changed or rebuild=True.
    """
    global _DB, _EMB

    if _DB is not None and not rebuild:
        return _DB

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    if _EMB is None:
        _EMB = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    if _need_rebuild(rebuild):
        docs = load_corpus()
        _DB = Chroma.from_documents(
            docs,
            _EMB,
            collection_name="krishi_rag",
            persist_directory=str(PERSIST_DIR),
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=CHROMA_SETTINGS,     # <- important
        )
        _save_meta()
        print(f"🧱 RAG index built: {len(docs)} chunks → {PERSIST_DIR}")
    else:
        _DB = Chroma(
            embedding_function=_EMB,
            collection_name="krishi_rag",
            persist_directory=str(PERSIST_DIR),
            client_settings=CHROMA_SETTINGS,     # <- important
        )
        print(f"📦 RAG index loaded from {PERSIST_DIR}")

    return _DB

def _find_seed_files() -> List[Path]:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    return [p for p in SEEDS_DIR.rglob("*") if p.suffix.lower() in {".pdf",".txt",".md"} and p.stat().st_size > 0]

def _fingerprint(files: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: x.as_posix()):
        st = p.stat()
        h.update(p.name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()

def _load_one(path: Path) -> List[Document]:
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        for d in pages:
            d.metadata["source"] = path.name
            d.metadata.setdefault("title", path.stem.replace("_"," ").title())
        return pages
    else:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = path.name
            d.metadata.setdefault("title", path.stem.replace("_"," ").title())
        return docs

def load_corpus() -> List[Document]:
    files = _find_seed_files()
    if not files:
        raise RuntimeError(f"No seed files found in {SEEDS_DIR}. Put PDFs/TXT there first.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n","\n"," ",""])
    docs: List[Document] = []
    for f in files:
        docs.extend(_load_one(f))
    return splitter.split_documents(docs)

def _need_rebuild(force: bool) -> bool:
    files = _find_seed_files()
    fp = _fingerprint(files)
    if force or not PERSIST_DIR.exists() or not META_PATH.exists():
        return True
    try:
        meta = json.loads(META_PATH.read_text())
        return meta.get("fingerprint") != fp
    except Exception:
        return True

def _save_meta():
    files = _find_seed_files()
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps({"fingerprint": _fingerprint(files)}, indent=2))

def build_or_load_index(rebuild: bool = False) -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    if _need_rebuild(rebuild):
        # fresh build
        docs = load_corpus()
        vectordb = Chroma.from_documents(
            docs, embeddings,
            collection_name="krishi_rag",
            persist_directory=str(PERSIST_DIR),
            collection_metadata={"hnsw:space":"cosine"}
        )
        _save_meta()
        return vectordb
    # load existing
    return Chroma(
        embedding_function=embeddings,
        collection_name="krishi_rag",
        persist_directory=str(PERSIST_DIR),
    )

def search(query: str, k: int = 4) -> List[Tuple[float, Document]]:
    db = build_or_load_index(rebuild=False)
    # returns (Document, score) with score ∈ [0,1], higher is better
    results = db.similarity_search_with_relevance_scores(query, k=k)
    # normalize to (similarity, Document)
    out = []
    for doc, score in results:
        out.append((float(score), doc))
    return out

# --------------- CLI ---------------
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Build/Query RAG index (Chroma + OpenAI)")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--ask", type=str, default=None)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    db = build_or_load_index(rebuild=args.rebuild)

    if args.ask:
        hits = search(args.ask, k=args.k)
        json_ready = []
        for sim, d in hits:
            m = d.metadata or {}
            json_ready.append({
                "similarity": round(sim, 3),
                "title": m.get("title"),
                "source": m.get("source"),
                "page": m.get("page"),
                "snippet": (d.page_content[:240] + "…") if d.page_content and len(d.page_content) > 240 else d.page_content,
            })
        print(json.dumps({"query": args.ask, "results": json_ready}, indent=2, ensure_ascii=False))
    else:
        print(f"Index ready: {PERSIST_DIR} (collection=krishi_rag)")


