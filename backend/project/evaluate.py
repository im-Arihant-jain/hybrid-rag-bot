import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pandas as pd
import mlflow
import dagshub

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
nli_model = pipeline("text-classification", model="roberta-large-mnli")

def normalize(t):
    return " ".join(t.lower().strip().split())

def exact_match(pred, gt):
    return 1.0 if normalize(pred) == normalize(gt) else 0.0

def f1(pred, gt):
    p = normalize(pred).split()
    g = normalize(gt).split()
    common = set(p) & set(g)
    if not common:
        return 0.0
    precision = len(common) / len(p)
    recall = len(common) / len(g)
    return 2 * (precision * recall) / (precision + recall)

def bleu(pred, gt):
    smooth = SmoothingFunction().method1
    return sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth)

def rougeL(pred, gt):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gt, pred)["rougeL"].fmeasure

def semantic_sim(pred, gt):
    a = semantic_model.encode([pred])
    b = semantic_model.encode([gt])
    return float(cosine_similarity(a, b)[0][0])

def bert_sc(pred, gt):
    P, R, F = bert_score([pred], [gt], lang="en", verbose=False)
    return float(F[0])



def evaluate_llm_predictions(
    predictions,
    ground_truths,
    queries,
    contexts=None,
    output_path="results.csv",
    mlflow_experiment="LLM-Evaluation"
):

    dagshub.init(
        repo_owner="arihantjain72000",
        repo_name="my-first-repo2",
        mlflow=True
    )

    mlflow.set_experiment(mlflow_experiment)

    results = []

    with mlflow.start_run():

        for i, (pred, gt, q) in enumerate(zip(predictions, ground_truths, queries)):

            context = contexts[i] if contexts else None

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
                
            }

            results.append(row)

        df = pd.DataFrame(results)

        # Compute averages
        avg_row = {
            "query": "AVERAGE",
            "prediction": "AVERAGE",
            "ground_truth": "AVERAGE",
            "exact_match": df["exact_match"].mean(),
            "f1": df["f1"].mean(),
            "bleu": df["bleu"].mean(),
            "rougeL": df["rougeL"].mean(),
            "semantic_similarity": df["semantic_similarity"].mean(),
            "bert_score": df["bert_score"].mean(), 
        }

        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        # Log metrics to MLflow
        mlflow.log_metrics({
            "avg_exact_match": avg_row["exact_match"],
            "avg_bleu": avg_row["bleu"],
            "avg_rougeL": avg_row["rougeL"],
            "avg_semantic_similarity": avg_row["semantic_similarity"],
            "avg_bert_score": avg_row["bert_score"]
        })

        # Save CSV artifact only for MLflow (not returned to frontend)
        df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)

    return df.to_dict(orient="records")
