import os
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from GlobalVars import MODEL, DEBUG, DB_DIR


# 🔧 Neo4j connection (set as environment variables)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://78116257.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password") 

graph_instance = None
def get_graph():
    global graph_instance
    if graph_instance is None:
        graph_instance = init_graph()
    return graph_instance

def init_graph():
    """
    Initialize connection to Neo4j.
    """
    if DEBUG:
        print("[GRAPH] Connecting to Neo4j...")
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    if DEBUG:
        print("[GRAPH] Connection successful.")
    return graph

def get_related_context(query: str, top_k: int = 5) -> str:
    """
    Retrieve up to top_k related document texts from Neo4j for the given query.
    Returns a concatenated string suitable for use as context in prompts.
    """
    if not query:
        return ""

    if DEBUG:
        print(f"[GRAPH] Getting related context for query: {query}")

    try:
        graph = get_graph()

        # Try whole-query text match first
        cypher = """
        MATCH (d:Document)
        WHERE toLower(d.text) CONTAINS toLower($q)
        RETURN d.text AS text
        LIMIT $limit
        """
        results = graph.query(cypher, params={"q": query, "limit": top_k})

        texts = []
        for r in results:
            if isinstance(r, dict):
                texts.append(r.get("text") or next(iter(r.values()), None))
            else:
                # fallback if result is a single value or tuple
                try:
                    texts.append(r[0])
                except Exception:
                    texts.append(str(r))

        # Fallback: search by tokens if no hits
        if not texts:
            tokens = [t for t in query.split() if len(t) > 3][:6]
            for term in tokens:
                res = graph.query(
                    """
                    MATCH (d:Document)
                    WHERE toLower(d.text) CONTAINS toLower($term)
                    RETURN d.text AS text
                    LIMIT $limit
                    """,
                    params={"term": term, "limit": top_k},
                )
                for r in res:
                    if isinstance(r, dict):
                        texts.append(r.get("text") or next(iter(r.values()), None))
                    else:
                        try:
                            texts.append(r[0])
                        except Exception:
                            texts.append(str(r))
                if len(texts) >= top_k:
                    break

        # Trim and join
        texts = [t for t in texts if t]
        context = "\n---\n".join(texts[:top_k]) if texts else ""
        if DEBUG:
            print(f"[GRAPH] Retrieved {len(texts[:top_k])} items for context.")
        return context

    except Exception as e:
        if DEBUG:
            print("[GRAPH] Error retrieving graph context:", e)
        return ""

def insert_docs_to_graph(graph, docs, embedding_model=MODEL):
    """
    Store document chunks as nodes in Neo4j with optional sequential relationships.
    """
    if DEBUG:
        print(f"[GRAPH] Inserting {len(docs)} documents into graph...")

    embeddings = OllamaEmbeddings(model=embedding_model)

    for i, doc in enumerate(docs):
        graph.query(
            """
            MERGE (d:Document {id: $id})
            SET d.text = $text
            """,
            params={"id": f"doc_{i}", "text": doc.page_content},
        )

        # Create NEXT relationships between chunks
        if i > 0:
            graph.query(
                """
                MATCH (a:Document {id: $id1}), (b:Document {id: $id2})
                MERGE (a)-[:NEXT]->(b)
                """,
                params={"id1": f"doc_{i-1}", "id2": f"doc_{i}"},
            )

    if DEBUG:
        print("[GRAPH] Document insertion complete.")


