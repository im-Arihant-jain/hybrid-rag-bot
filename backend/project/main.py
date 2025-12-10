
from fastapi import FastAPI, File, UploadFile, HTTPException
import mlflow
import dagshub
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from typing import List
import os
import shutil
import docProcess as docprocess
from query import chatapplicationApi
from fastapi.middleware.cors import CORSMiddleware 
from graphProcess import get_graph
from GlobalVars import FILE_DIR, DB_DIR, MODEL 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],  
)
if not os.path.exists(FILE_DIR):
    os.mkdir(FILE_DIR)
if not os.path.exists(DB_DIR):
    os.mkdir(DB_DIR)
def evaluate_llm_predictions(
    predictions, 
    ground_truths, 
    output_path="evaluation_results.csv",
    mlflow_experiment="LLM-Evaluation"
):

    # ----------------------------
    # 1. Connect to DagsHub MLflow
    # ----------------------------
    dagshub.init(repo_owner="arihantjain72000",
                 repo_name="my-first-repo2",
                 mlflow=True)
    
    mlflow.set_experiment(mlflow_experiment)

    results = []
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with mlflow.start_run():
        for pred, gt in zip(predictions, ground_truths):

            # Metrics
            exact_match = int(pred.strip() == gt.strip())
            bleu = sentence_bleu([gt.split()], pred.split())
            rouge_l = rouge.score(gt, pred)['rougeL'].fmeasure

            results.append({
                "prediction": pred,
                "ground_truth": gt,
                "exact_match": exact_match,
                "bleu": bleu,
                "rougeL": rouge_l
            })

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # --------------------------
        # Compute Averages
        # --------------------------
        avg_exact = df["exact_match"].mean()
        avg_bleu = df["bleu"].mean()
        avg_rouge = df["rougeL"].mean()

        # Log to MLflow
        mlflow.log_metric("avg_exact_match", avg_exact)
        mlflow.log_metric("avg_bleu", avg_bleu)
        mlflow.log_metric("avg_rougeL", avg_rouge)

        # ----------------------------------------
        # Add averages as a final row in the CSV
        # ----------------------------------------
        summary_row = {
            "prediction": "OVERALL_AVERAGE",
            "ground_truth": "OVERALL_AVERAGE",
            "exact_match": avg_exact,
            "bleu": avg_bleu,
            "rougeL": avg_rouge
        }

        df_with_summary = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        # Save output CSV/TXT
        if output_path.endswith(".csv"):
            df_with_summary.to_csv(output_path, index=False, sep="\t")
        elif output_path.endswith(".txt"):
            df_with_summary.to_csv(output_path, index=False, sep="\t")
        else:
            raise ValueError("Output file must be .csv or .txt")

        # Log artifact to MLflow
        mlflow.log_artifact(output_path)

        print(f"Evaluation completed. File saved at: {output_path}")

    return df_with_summary
@app.post("/uploadpdf/")
async def upload_pdf(file: UploadFile = File(...)):

    #uploading pdf file
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Read the file's contents
    contents = await file.read()
 
    
    save_path = os.path.join(FILE_DIR,file.filename) 
    print('filename :',save_path) 
    with open(save_path, "wb") as f:  
        f.write(contents)  
    file_path = os.path.join(FILE_DIR ,file.filename)
    document = docprocess.load_document(file_path)
    chunks = docprocess.split_document(document)

    client = docprocess.getclient(file.filename,MODEL,DB_DIR)
    docprocess.add_docs(client=client,chunks=chunks)

    del client
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents),
        "message": f"PDF file '{file.filename}' uploaded successfully!",
    }

           
@app.get('/getmetrics')
def getmetrics():
    preds = [
    "The capital of France is Paris.",
    "5 + 7 = 12"
    ]

    gts = [
        "The capital of France is Paris.",
        "5 + 7 = 12"
    ]

    evaluate_llm_predictions(preds, gts, output_path="results.csv")
    return {"status":"metrics logged successfully"}

@app.get('/getresult/{filename}/{query}')
def queryengine(filename:str,query:str):
    
    response = chatapplicationApi(query,filename,MODEL,DB_DIR)

    return response

@app.get('/delete')
def deletedata():
    if os.path.exists(FILE_DIR):
        for item in os.listdir(FILE_DIR):
            item_path = os.path.join(FILE_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):   
                os.remove(item_path)
            elif os.path.isdir(item_path):  
                shutil.rmtree(item_path)
    else:
        return {"error":f"something went wrong while deleting in {FILE_DIR}"}

    if os.path.exists(DB_DIR):
        for item in os.listdir(DB_DIR):
            item_path = os.path.join(DB_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):  
                os.remove(item_path)
            elif os.path.isdir(item_path):  
                shutil.rmtree(item_path) 
    else:
        return {"error":f"something went wrong while deleting in {DB_DIR}"}
    

    return {"status":"deleted successfully all the files and directories"}


