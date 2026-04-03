import os
import re
from typing import List

import spacy
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from GlobalVars import DEBUG, MODEL

load_dotenv()

# Load spaCy once for entity extraction during ingest.
nlp = spacy.load("en_core_web_sm")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://5e452542.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

SIMILARITY_THRESHOLD = 0.75
graph_instance = None


def get_graph():
    global graph_instance
    if graph_instance is None:
        graph_instance = init_graph()
    return graph_instance


def init_graph():
    if DEBUG:
        print("[GRAPH] Connecting to Neo4j...")
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    if DEBUG:
        print("[GRAPH] Connection successful.")
    return graph


def _normalize_texts(texts: List[str], top_k: int) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for text in texts:
        if not text:
            continue
        cleaned = " ".join(str(text).split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
        if len(ordered) >= top_k:
            break
    return ordered


def _query_text_rows(graph, cypher: str, params: dict) -> List[str]:
    rows = graph.query(cypher, params=params)
    texts: List[str] = []
    for row in rows:
        if isinstance(row, dict):
            texts.append(row.get("text") or next(iter(row.values()), None))
        else:
            try:
                texts.append(row[0])
            except Exception:
                texts.append(str(row))
    return texts


def get_related_context(query: str, top_k: int = 5, traversal_depth: int = 1) -> str:
    if not query:
        return ""

    depth = max(1, min(int(traversal_depth), 3))
    if DEBUG:
        print(f"[GRAPH] Retrieving context | top_k={top_k} | depth={depth}")

    try:
        graph = get_graph()

        seed_rows = graph.query(
            """
            MATCH (d:Document)
            WHERE toLower(d.text) CONTAINS toLower($q)
            RETURN d.id AS id, d.text AS text
            LIMIT $limit
            """,
            params={"q": query, "limit": top_k},
        )

        seed_ids: List[str] = []
        seed_texts: List[str] = []
        for row in seed_rows:
            if isinstance(row, dict):
                doc_id = row.get("id")
                text = row.get("text")
            else:
                doc_id = None
                text = str(row)
            if doc_id:
                seed_ids.append(doc_id)
            if text:
                seed_texts.append(text)

        # Fallback for sparse matches.
        if not seed_texts:
            terms = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(t) > 3][:8]
            for term in terms:
                seed_texts.extend(
                    _query_text_rows(
                        graph,
                        """
                        MATCH (d:Document)
                        WHERE toLower(d.text) CONTAINS toLower($term)
                        RETURN d.text AS text
                        LIMIT $limit
                        """,
                        {"term": term, "limit": top_k},
                    )
                )
                if len(seed_texts) >= top_k:
                    break

        expanded = list(seed_texts)
        if depth > 1 and seed_ids:
            expansion_cypher = f"""
            MATCH (d:Document)
            WHERE d.id IN $ids
            OPTIONAL MATCH (d)-[:NEXT|SIMILAR_TO*1..{depth}]-(nbr:Document)
            RETURN DISTINCT nbr.text AS text
            LIMIT $limit
            """
            expanded.extend(
                _query_text_rows(
                    graph, expansion_cypher, {"ids": seed_ids, "limit": top_k * depth}
                )
            )

            expanded.extend(
                _query_text_rows(
                    graph,
                    """
                    MATCH (d:Document)
                    WHERE d.id IN $ids
                    OPTIONAL MATCH (d)-[:HAS_ENTITY|BELONGS_TO_TOPIC]->(x)<-[:HAS_ENTITY|BELONGS_TO_TOPIC]-(nbr:Document)
                    RETURN DISTINCT nbr.text AS text
                    LIMIT $limit
                    """,
                    {"ids": seed_ids, "limit": top_k * depth},
                )
            )

        texts = _normalize_texts(expanded, top_k=top_k)
        return "\n---\n".join(texts) if texts else ""
    except Exception as e:
        if DEBUG:
            print("[GRAPH] Error retrieving graph context:", e)
        return ""


def get_topic_for_text(text: str) -> str:
    llm = ChatOllama(model=MODEL, temperature=0)
    prompt = f"""
    Read the following text and provide ONE short topic name (2-4 words).
    Text:
    {text}

    Respond with ONLY the topic label.
    """
    try:
        response = llm.invoke(prompt)
        topic = getattr(response, "content", str(response)).strip()
        return topic or "General"
    except Exception as e:
        if DEBUG:
            print("[GRAPH] Topic extraction failed:", e)
        return "General"


def insert_docs_to_graph(graph, docs, embedding_model=MODEL):
    if DEBUG:
        print(f"[GRAPH] Inserting {len(docs)} documents into graph...")

    embedder = OllamaEmbeddings(model=embedding_model)
    texts = [getattr(d, "page_content", str(d)) for d in docs]
    embeddings = embedder.embed_documents(texts)

    for i, text in enumerate(texts):
        vector = embeddings[i]

        graph.query(
            """
            MERGE (d:Document {id: $id})
            SET d.text = $text,
                d.embedding = $embedding
            """,
            params={"id": f"doc_{i}", "text": text, "embedding": vector},
        )

        if i > 0:
            graph.query(
                """
                MATCH (a:Document {id:$prev}), (b:Document {id:$curr})
                MERGE (a)-[:NEXT]->(b)
                """,
                params={"prev": f"doc_{i - 1}", "curr": f"doc_{i}"},
            )

        doc_nlp = nlp(text)
        for ent in doc_nlp.ents:
            graph.query(
                """
                MERGE (e:Entity {name:$name, type:$type})
                WITH e
                MATCH (d:Document {id:$doc_id})
                MERGE (d)-[:HAS_ENTITY]->(e)
                """,
                params={"name": ent.text, "type": ent.label_, "doc_id": f"doc_{i}"},
            )

        topic = get_topic_for_text(text)
        graph.query(
            """
            MERGE (t:Topic {name:$topic})
            WITH t
            MATCH (d:Document {id:$doc_id})
            MERGE (d)-[:BELONGS_TO_TOPIC]->(t)
            """,
            params={"topic": topic, "doc_id": f"doc_{i}"},
        )

    sim_matrix = cosine_similarity(embeddings)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sim_matrix[i][j] > SIMILARITY_THRESHOLD:
                graph.query(
                    """
                    MATCH (a:Document {id:$a}), (b:Document {id:$b})
                    MERGE (a)-[:SIMILAR_TO {score:$score}]->(b)
                    """,
                    params={
                        "a": f"doc_{i}",
                        "b": f"doc_{j}",
                        "score": float(sim_matrix[i][j]),
                    },
                )

    if DEBUG:
        print("[GRAPH] Document + semantic graph insertion complete.")
