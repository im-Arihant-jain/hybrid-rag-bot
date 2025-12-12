
from fastapi import FastAPI, File, UploadFile, HTTPException

import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from typing import List, Optional
import os
import shutil
import docProcess as docprocess
from query import chatapplicationApi
from fastapi.middleware.cors import CORSMiddleware 
from graphProcess import get_graph
from GlobalVars import FILE_DIR, DB_DIR, MODEL 
from evaluate import evaluate_llm_predictions

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

           
@app.post("/getmetrics")
def getmetrics(
    llm_outputs: List[str],
    ground_truths: List[str],
    queries: List[str],
    contexts: Optional[List[str]] = None
):
    results = evaluate_llm_predictions(
        predictions=llm_outputs,
        ground_truths=ground_truths,
        queries=queries,
        contexts=contexts,
        output_path="results.csv"
    )
    return {"status": "success", "metrics": results}

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


