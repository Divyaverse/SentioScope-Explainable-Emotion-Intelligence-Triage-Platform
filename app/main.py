import os
import time
import pickle
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.database import Base, engine, get_db
from app.models import QueryHistory
from app.schemas import (
    AnalyzeRequest, AnalyzeResponse, HealthResponse,
    HistoryItem, TriageInfo, WordAttribution
)
from app.normalizer import clean_and_normalize
from app.triage import compute_urgency
from app.explainer import FastLIMEExplainer, CLASS_NAMES

Base.metadata.create_all(bind=engine)

ml_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.path.join("artifacts", "bi_gru_model.keras")
    tokenizer_path = os.path.join("artifacts", "tokenizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        raise RuntimeError(f"Artifacts missing: Ensure {model_path} and {tokenizer_path} exist.")
        
    ml_components["model"] = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        ml_components["tokenizer"] = pickle.load(f)
        
    ml_components["explainer"] = FastLIMEExplainer(ml_components["model"], ml_components["tokenizer"])
    yield
    ml_components.clear()

app = FastAPI(
    title="SentioScope: Explainable Emotion AI Platform",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze_emotion(payload: AnalyzeRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    
    cleaned = clean_and_normalize(payload.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text contains no valid alphanumeric characters.")
        
    seq = ml_components["tokenizer"].texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=50, padding="post", truncating="post")
    
    preds = ml_components["model"].predict(padded, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = round(float(preds[top_idx]), 4)
    dominant_emotion = CLASS_NAMES[top_idx]
    
    probs = {CLASS_NAMES[i]: round(float(preds[i]), 4) for i in range(len(CLASS_NAMES))}
    triage_data = compute_urgency(probs)
    
    attributions = []
    if payload.explain:
        attributions = ml_components["explainer"].explain(cleaned)
        
    latency = round((time.time() - start_time) * 1000, 2)
    
    history_entry = QueryHistory(
        text=payload.text,
        dominant_emotion=dominant_emotion,
        confidence=confidence,
        urgency_tier=triage_data["tier"]
    )
    db.add(history_entry)
    db.commit()

    return AnalyzeResponse(
        original_text=payload.text,
        cleaned_text=cleaned,
        dominant_emotion=dominant_emotion,
        confidence=confidence,
        probabilities=probs,
        urgency=TriageInfo(**triage_data),
        word_attributions=[WordAttribution(**w) for w in attributions],
        latency_ms=latency
    )

@app.get("/api/v1/history", response_model=list[HistoryItem])
def get_history(limit: int = Query(20, le=100), db: Session = Depends(get_db)):
    rows = db.query(QueryHistory).order_by(QueryHistory.timestamp.desc()).limit(limit).all()
    return [
        HistoryItem(
            id=r.id,
            text=r.text,
            dominant_emotion=r.dominant_emotion,
            confidence=r.confidence,
            urgency_tier=r.urgency_tier,
            timestamp=r.timestamp.isoformat()
        ) for r in rows
    ]

@app.delete("/api/v1/history")
def clear_history(db: Session = Depends(get_db)):
    db.query(QueryHistory).delete()
    db.commit()
    return {"status": "History cleared"}

@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded="model" in ml_components,
        classes=CLASS_NAMES
    )
