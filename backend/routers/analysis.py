from fastapi import APIRouter, HTTPException
from models.schemas import AnalysisRequest
from services.market_data import get_yfinance_ticker, get_intraday_data, resample_data, calculate_moving_averages, calculate_technical_indicators
from services.gemini_service import ask_gemini_for_review
import pandas as pd

router = APIRouter()

@router.post("/review")
def analyze_review(req: AnalysisRequest):
    ticker = get_yfinance_ticker(req.ticker, req.market)
    if not ticker:
        raise HTTPException(status_code=404, detail="Ticker not found")
        
    # 데이터 로드
    df = get_intraday_data(ticker, req.market, '7d')
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
        
    # 만약 요청으로 들어온 분봉 간격이 있다면 적용
    # 1분봉, 3분봉 등 한글로 들어온다면 변환
    interval_map = {"1분봉": "1m", "3분봉": "3m", "5분봉": "5m", "10분봉": "10m"}
    interval = interval_map.get(req.interval_name, "1m")
    
    df = resample_data(df, interval)
    
    # 평단가 계산
    buy_dicts = [{"date": p.date, "price": p.price, "quantity": p.quantity} for p in req.buy_points]
    sell_dicts = [{"date": p.date, "price": p.price, "quantity": p.quantity} for p in req.sell_points]
    
    total_buy_qty = sum(p['quantity'] for p in buy_dicts)
    buy_avg = sum(p['price'] * p['quantity'] for p in buy_dicts) / total_buy_qty if total_buy_qty > 0 else 0
    
    total_sell_qty = sum(p['quantity'] for p in sell_dicts)
    sell_avg = sum(p['price'] * p['quantity'] for p in sell_dicts) / total_sell_qty if total_sell_qty > 0 else 0
    
    # 당일 데이터 필터링 (가장 첫 매수/매도 날짜 기준)
    if buy_dicts:
        target_date = pd.to_datetime(buy_dicts[0]['date']).date()
        df = df[df.index.date == target_date]
        
    if df.empty:
        raise HTTPException(status_code=400, detail=f"해당 날짜({target_date})의 차트 데이터가 없습니다.")
        
    result = ask_gemini_for_review(
        df=df,
        buy_points=buy_dicts,
        sell_points=sell_dicts,
        buy_avg_price=buy_avg,
        sell_avg_price=sell_avg,
        total_buy_quantity=total_buy_qty,
        total_sell_quantity=total_sell_qty,
        interval_name=req.interval_name
    )
    
    return result
