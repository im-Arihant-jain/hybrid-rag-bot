import re
import time
from typing import Dict, List, Tuple

from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from GlobalVars import *
from graphProcess import get_related_context

DEFAULT_MAX_REFLECTION_ITERATIONS = 2
DEFAULT_CONFIDENCE_TARGET = 0.72
DEFAULT_HALLUCINATION_MAX = 0.42
ESTIMATED_USD_PER_1K_TOKENS = 0.0002


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _estimate_tokens(text: str) -> int:
    return len(_tokenize(text))


def _adaptive_context_granularity(context: str, token_budget: int, complexity_label: str) -> str:
    if not context:
        return ""

    paragraphs = [p.strip() for p in context.split("\n") if p.strip()]
    if not paragraphs:
        return ""

    selected: List[str] = []
    used_tokens = 0
    max_budget = max(80, token_budget)

    if complexity_label == "complex":
        for para in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                t = _estimate_tokens(sent)
                if used_tokens + t > max_budget:
                    break
                selected.append(sent)
                used_tokens += t
            if used_tokens >= max_budget:
                break
    else:
        for para in paragraphs:
            t = _estimate_tokens(para)
            if used_tokens + t > max_budget:
                break
            selected.append(para)
            used_tokens += t

    return "\n".join(selected)


def analyze_query_complexity(query: str) -> Dict[str, object]:
    tokens = _tokenize(query)
    token_len = len(tokens)

    multi_hop_markers = {
        "compare", "relationship", "impact", "why", "how", "between",
        "connect", "before", "after", "cause", "effect", "depends",
        "derive", "link", "related", "difference", "across"
    }
    factual_markers = {"what", "when", "where", "who", "define", "list"}

    multi_hop_hits = sum(1 for t in tokens if t in multi_hop_markers)
    factual_hits = sum(1 for t in tokens if t in factual_markers)
    has_chain_punct = 1 if "?" in query and ("," in query or " and " in query.lower()) else 0

    score = min(
        1.0,
        (token_len / 35.0) * 0.35 + multi_hop_hits * 0.12 + factual_hits * 0.03 + has_chain_punct * 0.15,
    )

    if score < 0.35:
        label = "simple"
        graph_depth = 1
    elif score < 0.7:
        label = "medium"
        graph_depth = 2
    else:
        label = "complex"
        graph_depth = 3

    query_type = "multi_hop" if multi_hop_hits >= 2 or label == "complex" else "factual"
    return {
        "score": round(score, 4),
        "label": label,
        "graph_depth": graph_depth,
        "query_type": query_type,
        "token_len": token_len,
    }


def get_dynamic_retrieval_plan(complexity: Dict[str, object], iteration: int) -> Dict[str, int]:
    label = str(complexity["label"])
    base_vector_k = {"simple": 4, "medium": 7, "complex": 10}.get(label, 6)
    base_graph_k = {"simple": 3, "medium": 5, "complex": 8}.get(label, 4)

    bump = iteration * 2
    graph_depth = min(3, int(complexity["graph_depth"]) + (1 if iteration > 0 else 0))

    return {
        "vector_k": base_vector_k + bump,
        "graph_k": base_graph_k + bump,
        "graph_depth": graph_depth,
    }


def _vector_retrieval(db: Chroma, query: str, top_k: int) -> Tuple[List[Tuple[object, float]], str, float]:
    results = db.similarity_search_with_score(query, k=max(1, top_k))
    vector_context = "\n".join([doc.page_content for doc, _ in results if getattr(doc, "page_content", None)])

    if not results:
        return [], "", 0.0

    # Chroma returns distance-like scores; lower is better.
    confidences = [1.0 / (1.0 + max(float(score), 0.0)) for _, score in results]
    vector_confidence = sum(confidences) / len(confidences)
    return results, vector_context, float(vector_confidence)


def _dynamic_fusion_weights(
    complexity: Dict[str, object], vector_confidence: float, graph_context: str
) -> Dict[str, float]:
    label = str(complexity["label"])

    if label == "simple":
        vector_weight, graph_weight = 0.65, 0.35
    elif label == "medium":
        vector_weight, graph_weight = 0.55, 0.45
    else:
        vector_weight, graph_weight = 0.45, 0.55

    if vector_confidence < 0.45:
        graph_weight += 0.1
        vector_weight -= 0.1
    if not graph_context.strip():
        vector_weight, graph_weight = 0.9, 0.1

    total = vector_weight + graph_weight
    return {
        "vector": round(vector_weight / total, 4),
        "graph": round(graph_weight / total, 4),
    }


def _evidence_sufficiency(answer: str, context: str) -> float:
    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(context))
    if not answer_tokens or not context_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
    return float(max(0.0, min(1.0, overlap)))


def _hallucination_probability(answer: str, context: str, retrieval_confidence: float) -> float:
    suff = _evidence_sufficiency(answer, context)
    prob = 1.0 - (0.55 * suff + 0.45 * retrieval_confidence)
    return float(max(0.0, min(1.0, prob)))


def _response_confidence(retrieval_confidence: float, evidence_sufficiency: float, hallucination_probability: float) -> float:
    conf = 0.4 * retrieval_confidence + 0.4 * evidence_sufficiency + 0.2 * (1.0 - hallucination_probability)
    return float(max(0.0, min(1.0, conf)))


def _should_reflect(
    confidence: float, hallucination_probability: float, iteration: int, max_iterations: int
) -> bool:
    if iteration >= max_iterations:
        return False
    return confidence < DEFAULT_CONFIDENCE_TARGET or hallucination_probability > DEFAULT_HALLUCINATION_MAX


def chatapplicationApi(
    query: str,
    collectionName: str,
    modelName: str,
    dbpath: str,
    max_reflection_iterations: int = DEFAULT_MAX_REFLECTION_ITERATIONS,
):
    prompt_template = PromptTemplate.from_template(
        """
        You are a helpful AI assistant.
        Use the provided context to answer with factual grounding.
        If context is insufficient, explicitly mention uncertainty.

        ### Semantic (Vector) Context ###
        {vector_context}

        ### Graph Context (Relations / Entities) ###
        {graph_context}

        Question: {question}
        """
    )

    start_total = time.perf_counter()
    complexity = analyze_query_complexity(query)

    chat_model = ChatOllama(model=modelName, temperature=0)
    db = Chroma(
        collection_name=collectionName,
        embedding_function=OllamaEmbeddings(model=modelName),
        persist_directory=dbpath,
    )

    best = None
    iteration_logs: List[Dict[str, object]] = []
    total_tokens = 0

    for iteration in range(max(0, max_reflection_iterations) + 1):
        step_start = time.perf_counter()
        plan = get_dynamic_retrieval_plan(complexity, iteration)

        vector_results, raw_vector_context, vector_confidence = _vector_retrieval(
            db=db, query=query, top_k=plan["vector_k"]
        )
        raw_graph_context = get_related_context(
            query=query, top_k=plan["graph_k"], traversal_depth=plan["graph_depth"]
        )

        weights = _dynamic_fusion_weights(
            complexity=complexity,
            vector_confidence=vector_confidence,
            graph_context=raw_graph_context,
        )

        base_budget = {"simple": 450, "medium": 700, "complex": 950}.get(str(complexity["label"]), 650)
        budget = base_budget + iteration * 120
        vector_budget = int(budget * weights["vector"])
        graph_budget = max(80, budget - vector_budget)

        vector_context = _adaptive_context_granularity(
            raw_vector_context, token_budget=vector_budget, complexity_label=str(complexity["label"])
        )
        graph_context = _adaptive_context_granularity(
            raw_graph_context, token_budget=graph_budget, complexity_label=str(complexity["label"])
        )

        prompt = prompt_template.invoke(
            {
                "vector_context": vector_context,
                "graph_context": graph_context,
                "question": query,
            }
        )

        response = chat_model.invoke(prompt)
        answer = getattr(response, "content", str(response))
        fused_context = (vector_context + "\n" + graph_context).strip()

        evidence = _evidence_sufficiency(answer, fused_context)
        hallucination = _hallucination_probability(answer, fused_context, vector_confidence)
        confidence = _response_confidence(vector_confidence, evidence, hallucination)

        prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        prompt_tokens = _estimate_tokens(prompt_text)
        output_tokens = _estimate_tokens(answer)
        step_tokens = prompt_tokens + output_tokens
        total_tokens += step_tokens

        step_latency_ms = round((time.perf_counter() - step_start) * 1000.0, 2)
        iteration_log = {
            "iteration": iteration + 1,
            "plan": plan,
            "fusion_weights": weights,
            "retrieval_confidence": round(vector_confidence, 4),
            "evidence_sufficiency": round(evidence, 4),
            "hallucination_probability": round(hallucination, 4),
            "response_confidence": round(confidence, 4),
            "latency_ms": step_latency_ms,
            "token_usage_estimated": step_tokens,
        }
        iteration_logs.append(iteration_log)

        current = {
            "response": response,
            "answer": answer,
            "vector_results": vector_results,
            "vector_context": vector_context,
            "graph_context": graph_context,
            "plan": plan,
            "weights": weights,
            "vector_confidence": vector_confidence,
            "confidence": confidence,
            "evidence": evidence,
            "hallucination": hallucination,
        }

        if best is None or current["confidence"] > best["confidence"]:
            best = current

        if not _should_reflect(confidence, hallucination, iteration, max_reflection_iterations):
            best = current
            break

    del db

    total_latency_ms = round((time.perf_counter() - start_total) * 1000.0, 2)
    estimated_cost = round((total_tokens / 1000.0) * ESTIMATED_USD_PER_1K_TOKENS, 6)

    return {
        "content": best["answer"],
        "metadata": getattr(best["response"], "response_metadata", {}),
        "vector_context": best["vector_context"],
        "graph_context": best["graph_context"],
        "fused_context": (best["vector_context"] + "\n" + best["graph_context"]).strip(),
        "retrieved_from": {
            "vector_chunks": len(best["vector_results"]),
            "graph_context_len": len(best["graph_context"]),
            "graph_traversal_depth": best["plan"]["graph_depth"],
        },
        "adaptive": {
            "query_complexity": complexity,
            "final_fusion_weights": best["weights"],
            "reflection_iterations": len(iteration_logs),
            "iteration_logs": iteration_logs,
            "retrieval_confidence": round(best["vector_confidence"], 4),
            "evidence_sufficiency": round(best["evidence"], 4),
            "hallucination_probability": round(best["hallucination"], 4),
        },
        "runtime": {
            "latency_ms": total_latency_ms,
            "token_usage_estimated": total_tokens,
            "api_cost_estimate_usd": estimated_cost,
            "reflection_iterations": len(iteration_logs),
            "retrieval_confidence": round(best["vector_confidence"], 4),
            "evidence_sufficiency": round(best["evidence"], 4),
            "hallucination_probability": round(best["hallucination"], 4),
        },
    }


