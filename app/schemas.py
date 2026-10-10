from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, example="tbh idk why but I am terrified and feeling fml right now 😭")
    explain: bool = Field(default=True, description="Compute token-level saliency via LIME")

class WordAttribution(BaseModel):
    word: str
    weight: float

class TriageInfo(BaseModel):
    score: float
    tier: str

class AnalyzeResponse(BaseModel):
    original_text: str
    cleaned_text: str
    dominant_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    urgency: TriageInfo
    word_attributions: List[WordAttribution]
    latency_ms: float

class HistoryItem(BaseModel):
    id: int
    text: str
    dominant_emotion: str
    confidence: float
    urgency_tier: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: List[str]
