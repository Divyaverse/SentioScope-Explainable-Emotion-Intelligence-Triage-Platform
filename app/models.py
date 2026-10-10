from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String
from app.database import Base

class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text = Column(String, nullable=False)
    dominant_emotion = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    urgency_tier = Column(String, nullable=False)
