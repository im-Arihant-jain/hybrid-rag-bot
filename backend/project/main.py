import math
import os
import shutil
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import docProcess as docprocess
from evaluate import evaluate_llm_predictions
from GlobalVars import DB_DIR, FILE_DIR, MODEL
from query import chatapplicationApi

app = FastAPI()
contexts: List[str] = []

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
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    contents = await file.read()
    save_path = os.path.join(FILE_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    document = docprocess.load_document(save_path)
    chunks = docprocess.split_document(document)
    client = docprocess.getclient(file.filename, MODEL, DB_DIR)
    docprocess.add_docs(client=client, chunks=chunks)
    del client

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents),
        "message": f"PDF file '{file.filename}' uploaded successfully!",
    }


class RuntimeMetric(BaseModel):
    latency_ms: Optional[float] = None
    token_usage_estimated: Optional[float] = None
    api_cost_estimate_usd: Optional[float] = None
    reflection_iterations: Optional[int] = None
    retrieval_confidence: Optional[float] = None
    evidence_sufficiency: Optional[float] = None
    hallucination_probability: Optional[float] = None


class MetricsRequest(BaseModel):
    llm_outputs: List[str]
    ground_truths: List[str]
    queries: List[str]
    runtime_metrics: Optional[List[RuntimeMetric]] = None


def getmetrics_validate_shape(req: MetricsRequest):
    count = len(req.llm_outputs)
    if not (len(req.ground_truths) == count and len(req.queries) == count):
        raise HTTPException(status_code=400, detail="All input lists must have the same length")
    if req.runtime_metrics is not None and len(req.runtime_metrics) != count:
        raise HTTPException(
            status_code=400,
            detail="runtime_metrics length must match llm_outputs length when provided",
        )


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


@app.post("/getmetrics")
def getmetrics(req: MetricsRequest):
    getmetrics_validate_shape(req)
    runtime_metrics = [m.model_dump() for m in req.runtime_metrics] if req.runtime_metrics else None
    results = evaluate_llm_predictions(
        predictions=req.llm_outputs,
        ground_truths=req.ground_truths,
        queries=req.queries,
        contexts=contexts,
        runtime_metrics=runtime_metrics,
        output_path="results.csv",
    )
    return {"status": "success", "metrics": _sanitize_for_json(results)}


@app.get("/getresult/{filename}/{query}")
def queryengine(filename: str, query: str):
    response = chatapplicationApi(query, filename, MODEL, DB_DIR)
    evidence_context = response.get("fused_context") or (
        response.get("vector_context", "") + "\n" + response.get("graph_context", "")
    )
    contexts.append(evidence_context)
    if len(contexts) > 1000:
        del contexts[0]
    return _sanitize_for_json(response)


@app.get("/delete")
def deletedata():
    if os.path.exists(FILE_DIR):
        for item in os.listdir(FILE_DIR):
            item_path = os.path.join(FILE_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        return {"error": f"something went wrong while deleting in {FILE_DIR}"}

    if os.path.exists(DB_DIR):
        for item in os.listdir(DB_DIR):
            item_path = os.path.join(DB_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        return {"error": f"something went wrong while deleting in {DB_DIR}"}

    return {"status": "deleted successfully all the files and directories"}


