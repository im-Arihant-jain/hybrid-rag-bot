from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from GlobalVars import *
from graphProcess import get_graph, insert_docs_to_graph

import os       
import shutil    


def load_document(file_path):
    """
    Load a PDF file as LangChain documents.
    """
    if DEBUG:
        print(f"[DOC] Loading document from: {file_path}")

    document_loader = PyPDFLoader(file_path)
    return document_loader.load()


def split_document(document):
    """
    Split document into overlapping chunks.
    """
    if DEBUG:
        print(f"[DOC] Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
    )
    chunks = splitter.split_documents(document)

    if DEBUG:
        print(f"[DOC] Total chunks created: {len(chunks)}")

    return chunks


def get_embeddings_function(model_name):
    return OllamaEmbeddings(model=model_name)


def create_embedding(chunks, modelname: str):
    """
    Create embeddings for chunks using Ollama model.
    """
    if DEBUG:
        print(f"[DOC] Creating embeddings using model: {modelname}")

    embedding_model = get_embeddings_function(modelname)
    return embedding_model.embed_documents(chunks)


def add_docs(client, chunks):
    """
    Add document chunks to Chroma DB and sync them to Neo4j.
    """
    if DEBUG:
        print(f"[DOC] Adding {len(chunks)} chunks to Chroma DB...")

    client.add_documents(chunks)

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
    """
    Get or create a Chroma collection client.
    """
    if DEBUG:
        print(f"[DOC] Getting Chroma client for: {collection_name}")

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings_function(model_name),
        persist_directory=directory
    )


def erase(FILE_DIR, DB_DIR):
    """
    Clears all files and directories inside FILE_DIR and DB_DIR.
    """
    if DEBUG:
        print(f"[DOC] Clearing FILE_DIR: {FILE_DIR}")
        print(f"[DOC] Clearing DB_DIR: {DB_DIR}")

    # --- Clear uploaded files ---
    if os.path.exists(FILE_DIR):
        for item in os.listdir(FILE_DIR):
            item_path = os.path.join(FILE_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        return {"error": f"Something went wrong while deleting in {FILE_DIR}"}

    # --- Clear Chroma DB ---
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
