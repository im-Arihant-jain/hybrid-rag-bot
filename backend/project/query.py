# query.py (Hybrid Graph RAG)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma.vectorstores import Chroma
from GlobalVars import *
from graphProcess import get_related_context   

def chatapplicationApi(query: str, collectionName: str, modelName: str, dbpath: str):
    """
    Hybrid retrieval pipeline:
      1. Retrieve semantic matches from Chroma (vector similarity)
      2. Retrieve related context from Neo4j (graph-based)
      3. Fuse both contexts for richer answers
    """
    # --- Prompt Template ---
    prompt_template = PromptTemplate.from_template(
        """
        You are a helpful AI assistant.
        Answer the user's question using the following two sources of context.

        ### Semantic (Vector) Context ###
        {vector_context}

        ### Graph Context (Relationships / Entities) ###
        {graph_context}

        Use both contexts to provide a well-structured and complete answer.

        Question: {question}
        """
    )

    # --- Models ---
    chatModel = ChatOllama(model=modelName)
    db = Chroma(
        collection_name=collectionName,
        embedding_function=OllamaEmbeddings(model=modelName),
        persist_directory=dbpath
    )

    # --- Vector Retrieval (from Chroma) ---
    results = db.similarity_search_with_score(query)
    vector_context = "\n".join([doc.page_content for doc, score in results])

    # --- Graph Retrieval (from Neo4j) ---
    graph_context = get_related_context(query)

    # --- Build hybrid prompt ---
    prompt = prompt_template.invoke({
        "vector_context": vector_context,
        "graph_context": graph_context,
        "question": query
    })

    # --- Generate answer using Ollama ---
    response = chatModel.invoke(prompt)

    del db
    return {
        "content": response.content,
        "metadata": response.response_metadata,
        "retrieved_from": {
            "vector_chunks": len(results),
            "graph_context_len": len(graph_context)
        }
    }
