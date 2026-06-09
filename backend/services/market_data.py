import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
import pytz

def get_all_stocks_kr():
    try:
        return fdr.StockListing('KRX')
    except:
        return pd.DataFrame()

def get_yfinance_ticker(stock_name: str, market: str) -> str:
    if market == "US":
        # For US, stock_name is already ticker
        return stock_name.upper()

    # For KR, use FDR to find it
    df_krx = get_all_stocks_kr()
    if df_krx.empty:
        return ""
    
    stock_name_lower = stock_name.lower()
    df_krx['Name_lower'] = df_krx['Name'].str.lower()
    exact_match = df_krx[df_krx['Name_lower'] == stock_name_lower]
    if exact_match.empty:
        return ""
    
    stock_info = exact_match.iloc[0]
    stock_code = stock_info['Code']
    krx_market = stock_info['Market']
    
    if krx_market == 'KOSPI':
        return f"{stock_code}.KS"
    else:
        return f"{stock_code}.KQ"

def get_stock_suggestions(search_term: str) -> list:
    df_krx = get_all_stocks_kr()
    if df_krx.empty or len(search_term) < 1:
        return []
    
    search_term_lower = search_term.lower()
    df_krx['Name_lower'] = df_krx['Name'].str.lower()
    matches = df_krx[df_krx['Name_lower'].str.contains(search_term_lower, na=False)]
    matches = matches.sort_values('Name')
    return matches['Name'].tolist()[:20]

def resample_data(data: pd.DataFrame, resample_rule: str):
    if not resample_rule or resample_rule == "1m":
        return data
    
    resample_map = {"3m": "3T", "5m": "5T", "10m": "10T"}
    rule = resample_map.get(resample_rule)
    if not rule:
        return data

    try:
        resampled = data.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return resampled
    except:
        return data

def calculate_moving_averages(df: pd.DataFrame, periods=[5, 10, 20, 60, 120]):
    for period in periods:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
    return df

def calculate_technical_indicators(df: pd.DataFrame):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    if 'MA_20' in df.columns:
        df['Price_MA20_Deviation'] = ((df['Close'] - df['MA_20']) / df['MA_20']) * 100
        
    df['Volume_Ratio'] = (df['Volume'] / df['Volume'].rolling(window=20).mean()) * 100
    
    return df

def get_intraday_data(ticker: str, market: str, period='7d') -> pd.DataFrame:
    try:
        # Always download 1m first to allow flexible resampling
        data = yf.download(tickers=ticker, period=period, interval='1m', prepost=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        if data.empty:
            return pd.DataFrame()
            
        if data.index.tz is not None:
            data.index = data.index.tz_convert('UTC')
        else:
            data.index = data.index.tz_localize('UTC')
            
        if market == "KR":
            data.index = data.index.tz_convert('Asia/Seoul')
            data = data.between_time('09:00', '15:30')
        elif market == "US":
            data.index = data.index.tz_convert('America/New_York')
            data = data.between_time('09:30', '16:00')
            
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return pd.DataFrame()
