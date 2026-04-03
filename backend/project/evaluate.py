import contextlib
from typing import Dict, List, Optional

import dagshub
import mlflow
import pandas as pd
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def exact_match(pred: str, gt: str) -> float:
    return 1.0 if normalize(pred) == normalize(gt) else 0.0


def _tokens(text: str) -> List[str]:
    return normalize(text).split()


def lexical_overlap(pred: str, ctx: str) -> float:
    p = set(_tokens(pred))
    c = set(_tokens(ctx))
    if not p or not c:
        return 0.0
    return len(p & c) / max(1, len(p))


def context_semantic_similarity(pred: str, context: Optional[str]) -> float:
    if context is None or context.strip() == "":
        return 0.0
    a = semantic_model.encode([pred or ""])
    b = semantic_model.encode([context])
    return float(cosine_similarity(a, b)[0][0])


def context_bert_score(pred: str, context: Optional[str]) -> float:
    if context is None or context.strip() == "":
        return 0.0
    _, _, f1 = bert_score([pred or ""], [context], lang="en", verbose=False)
    return float(f1[0])


def evidence_sufficiency(pred: str, context: Optional[str]) -> float:
    if context is None or context.strip() == "":
        return 0.0
    sem = context_semantic_similarity(pred, context)
    overlap = lexical_overlap(pred, context)
    score = 0.6 * sem + 0.4 * overlap
    return float(max(0.0, min(1.0, score)))


def hallucination_probability(pred: str, context: Optional[str]) -> float:
    if context is None or context.strip() == "":
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - evidence_sufficiency(pred, context))))


def estimate_token_usage(pred: str, context: Optional[str]) -> int:
    context_tokens = len(_tokens(context or ""))
    return len(_tokens(pred or "")) + context_tokens


def estimate_api_cost(token_usage: float, usd_per_1k_tokens: float = 0.0002) -> float:
    return round((float(token_usage) / 1000.0) * usd_per_1k_tokens, 6)


def f1(pred: str, gt: str) -> float:
    p = _tokens(pred)
    g = _tokens(gt)
    common = set(p) & set(g)
    if not common:
        return 0.0
    precision = len(common) / len(p)
    recall = len(common) / len(g)
    return 2 * (precision * recall) / (precision + recall)


def bleu(pred: str, gt: str) -> float:
    smooth = SmoothingFunction().method1
    return sentence_bleu([_tokens(gt)], _tokens(pred), smoothing_function=smooth)


def rougeL(pred: str, gt: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gt or "", pred or "")["rougeL"].fmeasure


def semantic_sim(pred: str, gt: str) -> float:
    a = semantic_model.encode([pred or ""])
    b = semantic_model.encode([gt or ""])
    return float(cosine_similarity(a, b)[0][0])


def bert_sc(pred: str, gt: str) -> float:
    _, _, f1 = bert_score([pred or ""], [gt or ""], lang="en", verbose=False)
    return float(f1[0])


def evaluate_llm_predictions(
    predictions: List[str],
    ground_truths: List[str],
    queries: List[str],
    contexts: Optional[List[str]] = None,
    runtime_metrics: Optional[List[Dict[str, object]]] = None,
    output_path: str = "results.csv",
    mlflow_experiment: str = "LLM-Evaluation",
):
    tracking_enabled = True
    run_context = contextlib.nullcontext()

    try:
        dagshub.init(repo_owner="arihantjain72000", repo_name="my-first-repo2", mlflow=True)
        mlflow.set_experiment(mlflow_experiment)
        run_context = mlflow.start_run()
    except Exception as e:
        tracking_enabled = False
        print(f"[EVAL][WARN] MLflow/DagsHub tracking disabled: {e}")

    results = []
    with run_context:
        for i, (pred, gt, q) in enumerate(zip(predictions, ground_truths, queries)):
            context = contexts[i] if contexts and i < len(contexts) else None
            runtime = runtime_metrics[i] if runtime_metrics and i < len(runtime_metrics) else {}

            runtime_token_usage = runtime.get("token_usage_estimated")
            runtime_latency = runtime.get("latency_ms")
            runtime_reflections = runtime.get("reflection_iterations")
            runtime_retrieval_conf = runtime.get("retrieval_confidence")
            runtime_evidence = runtime.get("evidence_sufficiency")
            runtime_hallucination = runtime.get("hallucination_probability")

            token_usage = (
                float(runtime_token_usage)
                if runtime_token_usage is not None
                else float(estimate_token_usage(pred, context))
            )
            evidence = (
                float(runtime_evidence)
                if runtime_evidence is not None
                else float(evidence_sufficiency(pred, context))
            )
            halluc_prob = (
                float(runtime_hallucination)
                if runtime_hallucination is not None
                else float(hallucination_probability(pred, context))
            )

            api_cost = runtime.get("api_cost_estimate_usd")
            if api_cost is None:
                api_cost = estimate_api_cost(token_usage)

            row = {
                "query": q,
                "prediction": pred,
                "ground_truth": gt,
                "exact_match": exact_match(pred, gt),
                "f1": f1(pred, gt),
                "bleu": bleu(pred, gt),
                "rougeL": rougeL(pred, gt),
                "semantic_similarity": semantic_sim(pred, gt),
                "bert_score": bert_sc(pred, gt),
                "context_semantic_similarity": context_semantic_similarity(pred, context),
                "context_bert_score": context_bert_score(pred, context),
                "evidence_sufficiency": evidence,
                "hallucination_probability": halluc_prob,
                "token_usage": token_usage,
                "latency_ms": float(runtime_latency) if runtime_latency is not None else None,
                "api_cost_estimate_usd": float(api_cost),
                "reflection_iterations": int(runtime_reflections) if runtime_reflections is not None else 1,
                "retrieval_confidence": float(runtime_retrieval_conf) if runtime_retrieval_conf is not None else None,
            }
            results.append(row)

        df = pd.DataFrame(results)

        numeric_cols = [
            "exact_match",
            "f1",
            "bleu",
            "rougeL",
            "semantic_similarity",
            "bert_score",
            "context_semantic_similarity",
            "context_bert_score",
            "evidence_sufficiency",
            "hallucination_probability",
            "token_usage",
            "latency_ms",
            "api_cost_estimate_usd",
            "reflection_iterations",
            "retrieval_confidence",
        ]
        averages = {col: pd.to_numeric(df[col], errors="coerce").mean() for col in numeric_cols}

        avg_row = {
            "query": "AVERAGE",
            "prediction": "AVERAGE",
            "ground_truth": "AVERAGE",
            **averages,
        }

        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        df.to_csv(output_path, index=False)

        if tracking_enabled:
            mlflow.log_metrics(
                {
                    "avg_exact_match": float(averages["exact_match"]),
                    "avg_bleu": float(averages["bleu"]),
                    "avg_f1": float(averages["f1"]),
                    "avg_rougeL": float(averages["rougeL"]),
                    "avg_semantic_similarity": float(averages["semantic_similarity"]),
                    "avg_bert_score": float(averages["bert_score"]),
                    "avg_context_semantic_similarity": float(averages["context_semantic_similarity"]),
                    "avg_context_bert_score": float(averages["context_bert_score"]),
                    "avg_evidence_sufficiency": float(averages["evidence_sufficiency"]),
                    "avg_hallucination_probability": float(averages["hallucination_probability"]),
                    "avg_token_usage": float(averages["token_usage"]),
                    "avg_latency_ms": float(averages["latency_ms"])
                    if pd.notna(averages["latency_ms"])
                    else 0.0,
                    "avg_api_cost_estimate_usd": float(averages["api_cost_estimate_usd"]),
                }
            )
            mlflow.log_artifact(output_path)

    return df.to_dict(orient="records")
