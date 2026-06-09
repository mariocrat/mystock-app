from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class TradePoint(BaseModel):
    date: str # e.g. "2024-05-13T09:30:00"
    price: float
    quantity: int

class MarketDataRequest(BaseModel):
    ticker: str
    market: str # "KR" or "US"
    interval: str = "1m"
    period: str = "7d"

class AnalysisRequest(BaseModel):
    ticker: str
    market: str
    buy_points: List[TradePoint]
    sell_points: List[TradePoint]
    interval_name: str = "1분봉"
