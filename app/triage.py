def compute_urgency(probabilities: dict) -> dict:
    sadness_w = probabilities.get("sadness", 0.0) * 0.85
    fear_w = probabilities.get("fear", 0.0) * 0.75
    anger_w = probabilities.get("anger", 0.0) * 0.60
    
    raw_score = (sadness_w + fear_w + anger_w) * 100.0
    urgency_score = min(100.0, round(raw_score, 1))

    if urgency_score >= 60.0:
        tier = "CRITICAL"
    elif urgency_score >= 30.0:
        tier = "ELEVATED"
    else:
        tier = "NORMAL"

    return {"score": urgency_score, "tier": tier}
