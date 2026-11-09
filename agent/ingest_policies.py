# agent/ingest_policies.py   (run with: python agent/ingest_policies.py)
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "policies"
STORE = ROOT / "chroma_db"
COLLECTION = "policies"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    files = sorted(POLICIES.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files in {POLICIES}")
    docs = [Document(page_content=f.read_text(), metadata={"source": f.name}) for f in files]
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # Rebuild from scratch so reruns do not duplicate chunks.
    Chroma(collection_name=COLLECTION, persist_directory=str(STORE), embedding_function=emb).delete_collection()
    Chroma.from_documents(chunks, emb, collection_name=COLLECTION, persist_directory=str(STORE))
    print(f"Stored {len(chunks)} chunks from {len(files)} policy files in {STORE}")


if __name__ == "__main__":
    main()
