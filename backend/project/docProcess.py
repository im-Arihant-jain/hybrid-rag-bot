import os
import shutil
from typing import Dict

from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from GlobalVars import *
from graphProcess import get_graph, insert_docs_to_graph


def load_document(file_path):
    if DEBUG:
        print(f"[DOC] Loading document from: {file_path}")
    document_loader = PyPDFLoader(file_path)
    return document_loader.load()


def infer_document_profile(document) -> Dict[str, object]:
    page_count = len(document)
    lengths = [len(getattr(d, "page_content", "")) for d in document] if document else [0]
    total_chars = sum(lengths)
    avg_chars = total_chars / max(1, page_count)

    if total_chars <= 5000:
        profile = "short"
    elif page_count >= 25 or avg_chars >= 3500:
        profile = "dense"
    else:
        profile = "standard"

    return {
        "profile": profile,
        "page_count": page_count,
        "total_chars": total_chars,
        "avg_chars_per_page": round(avg_chars, 2),
    }


def get_chunk_params(profile: str) -> Dict[str, int]:
    if profile == "short":
        return {"chunk_size": 1400, "chunk_overlap": 180}
    if profile == "dense":
        return {"chunk_size": 800, "chunk_overlap": 200}
    return {"chunk_size": 1024, "chunk_overlap": 220}


def split_document(document, chunk_strategy: str = "adaptive", chunk_size: int = None, chunk_overlap: int = None):
    if DEBUG:
        print("[DOC] Splitting document into chunks...")

    profile_info = infer_document_profile(document)
    selected_profile = profile_info["profile"]

    if chunk_strategy == "adaptive" and (chunk_size is None or chunk_overlap is None):
        params = get_chunk_params(selected_profile)
        chunk_size = params["chunk_size"]
        chunk_overlap = params["chunk_overlap"]
    else:
        chunk_size = chunk_size or 1024
        chunk_overlap = chunk_overlap or 220

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(document)

    for idx, chunk in enumerate(chunks):
        metadata = getattr(chunk, "metadata", {}) or {}
        metadata.update(
            {
                "chunk_index": idx,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_profile": selected_profile,
            }
        )
        chunk.metadata = metadata

    if DEBUG:
        print(
            f"[DOC] Profile={selected_profile} | chunk_size={chunk_size} | chunk_overlap={chunk_overlap} | chunks={len(chunks)}"
        )

    return chunks


def get_embeddings_function(model_name):
    return OllamaEmbeddings(model=model_name)


def add_docs(client, chunks):
    if DEBUG:
        print(f"[DOC] Adding {len(chunks)} chunks to Chroma DB...")

    texts = [getattr(c, "page_content", str(c)) for c in chunks]
    metadatas = [getattr(c, "metadata", {}) or {} for c in chunks]
    client.add_texts(texts=texts, metadatas=metadatas)

    if DEBUG:
        print("[DOC] Syncing documents to Neo4j...")
    try:
        graph = get_graph()
        insert_docs_to_graph(graph, chunks, embedding_model=MODEL)
        if DEBUG:
            print("[DOC] Neo4j sync completed successfully.")
    except Exception as e:
        print(f"[DOC][ERROR] Failed to sync with Neo4j: {e}")


def getclient(collection_name, model_name, directory):
    if DEBUG:
        print(f"[DOC] Getting Chroma client for: {collection_name}")
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings_function(model_name),
        persist_directory=directory,
    )


def erase(FILE_DIR, DB_DIR):
    if DEBUG:
        print(f"[DOC] Clearing FILE_DIR: {FILE_DIR}")
        print(f"[DOC] Clearing DB_DIR: {DB_DIR}")

    if os.path.exists(FILE_DIR):
        for item in os.listdir(FILE_DIR):
            item_path = os.path.join(FILE_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        return {"error": f"Something went wrong while deleting in {FILE_DIR}"}

    if os.path.exists(DB_DIR):
        for item in os.listdir(DB_DIR):
            item_path = os.path.join(DB_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        return {"error": f"Something went wrong while deleting in {DB_DIR}"}

    if DEBUG:
        print("[DOC] Data directories cleared successfully.")
