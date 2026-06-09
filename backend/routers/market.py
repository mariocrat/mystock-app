from fastapi import APIRouter, HTTPException
from typing import List
from models.schemas import MarketDataRequest
from services.market_data import get_stock_suggestions, get_yfinance_ticker, get_intraday_data, resample_data, calculate_moving_averages, calculate_technical_indicators
import json

router = APIRouter()

@router.get("/search")
def search_stock(q: str):
    suggestions = get_stock_suggestions(q)
    return {"suggestions": suggestions}

@router.post("/data")
def get_chart_data(req: MarketDataRequest):
    ticker = get_yfinance_ticker(req.ticker, req.market)
    if not ticker:
        raise HTTPException(status_code=404, detail="Ticker not found")
        
    df = get_intraday_data(ticker, req.market, req.period)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found from yfinance")
        
    df = resample_data(df, req.interval)
    df = calculate_moving_averages(df)
    df = calculate_technical_indicators(df)
    
    import numpy as np
    import json
    
    # NaN 및 inf를 numpy NaN으로 우선 통일
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # TradingView 호환을 위해 컬럼명 추가 매핑
    df['time'] = df.index.map(lambda x: x.isoformat())
    df['open'] = df['Open']
    df['high'] = df['High']
    df['low'] = df['Low']
    df['close'] = df['Close']
    df['value'] = df['Volume']
    
    # pandas to_json을 통해 모든 numpy.int64, numpy.float64를 네이티브 타입으로 안전하게 파싱 (NaN은 자동 null 처리됨)
    parsed_json = df.to_json(orient="records", date_format="iso")
    data_list = json.loads(parsed_json)
        
    return {
        "ticker": ticker,
        "market": req.market,
        "interval": req.interval,
        "data": data_list
    }
