import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import random
import yfinance as yf
import google.generativeai as genai
import json
import FinanceDataReader as fdr

@st.cache_data
def get_krx_stock_list():
    # 한국 주식 전체 목록을 한 번만 불러와서 캐싱합니다.
    return fdr.StockListing('KRX')

# 페이지 기본 설정
st.set_page_config(
    page_title="AI 주식 트레이닝 웹 앱",
    page_icon="📈",
    layout="wide"
)

# 앱 제목
st.title("📈 AI 주식 트레이닝 웹 앱")
st.markdown("---")

# 사이드바에 메뉴 생성
st.sidebar.title("메뉴 선택")
menu = st.bar.selectbox(
    "원하는 기능을 선택하세요:",
    ["내 매매 타점 복기", "차트 트레이닝 퀴즈"]
)

# 전역 변수로 종목 리스트 캐싱
@st.cache_data
def get_all_stocks():
    """
    KRX 전체 종목 리스트를 가져오는 함수 (캐싱 적용)
    """
    try:
        df_krx = fdr.StockListing('KRX')
        return df_krx
    except:
        return pd.DataFrame()

def get_stock_code(stock_name):
    """
    종목명(한글)으로 종목 코드를 찾는 함수 (대소문자 구별 없이)
    """
    try:
        df_krx = get_all_stocks()
        if df_krx.empty:
            return None
        
        # 대소문자 구별 없이 검색
        stock_name_lower = stock_name.lower()
        df_krx['Name_lower'] = df_krx['Name'].str.lower()
        
        # 정확히 일치하는 종목 찾기
        exact_match = df_krx[df_krx['Name_lower'] == stock_name_lower]
        if not exact_match.empty:
            return exact_match['Code'].values[0]
        
        # 부분 일치하는 종목들 반환 (드롭다운용)
        partial_matches = df_krx[df_krx['Name_lower'].str.contains(stock_name_lower, na=False)]
        if not partial_matches.empty:
            return partial_matches.iloc[0]['Code']  # 첫 번째 매치 반환
        
        return None
    except:
        return None

def get_yfinance_ticker(stock_name):
    """
    한글 종목명을 yfinance 티커로 변환하는 함수 (FDR + yfinance 하이브리드)
    """
    try:
        df_krx = get_all_stocks()
        if df_krx.empty:
            return None
        
        # 대소문자 구별 없이 검색
        stock_name_lower = stock_name.lower()
        df_krx['Name_lower'] = df_krx['Name'].str.lower()
        
        # 정확히 일치하는 종목 찾기
        exact_match = df_krx[df_krx['Name_lower'] == stock_name_lower]
        if exact_match.empty:
            return None
        
        # 종목 정보 추출
        stock_info = exact_match.iloc[0]
        stock_code = stock_info['Code']
        market = stock_info['Market']  # KOSPI, KOSDAQ 등
        
        # 시장에 따라 접미사 추가
        if market == 'KOSPI':
            ticker = f"{stock_code}.KS"
        elif market == 'KOSDAQ':
            ticker = f"{stock_code}.KQ"
        else:
            # 다른 시장의 경우 KOSDAQ으로 처리
            ticker = f"{stock_code}.KQ"
        
        return ticker
    except Exception as e:
        st.error(f"티커 변환 오류: {str(e)}")
        return None

def get_stock_suggestions(search_term):
    """
    검색어에 맞는 종목명 제안 리스트 반환
    """
    try:
        df_krx = get_all_stocks()
        if df_krx.empty or len(search_term) < 1:
            return []
        
        search_term_lower = search_term.lower()
        df_krx['Name_lower'] = df_krx['Name'].str.lower()
        
        # 부분 일치하는 종목들 찾기 (가나다 순 정렬)
        matches = df_krx[df_krx['Name_lower'].str.contains(search_term_lower, na=False)]
        matches = matches.sort_values('Name')
        
        return matches['Name'].tolist()[:20]  # 최대 20개만 반환
    except:
        return []

def resample_data(data, resample_rule):
    """
    Pandas Resampling으로 분봉 데이터 변환
    resample_rule: None (1분봉), '3T' (3분봉), '5T' (5분봉), '10T' (10분봉)
    """
    if resample_rule is None:
        return data  # 1분봉은 그대로 반환
    
    try:
        # OHLCV 리샘플링
        resampled = data.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    except Exception as e:
        st.error(f"데이터 리샘플링 중 오류 발생: {str(e)}")
        return data

def get_intraday_data(ticker, interval='1m', period='7d'):
    """
    yfinance를 사용해 분봉 데이터를 가져오는 함수
    interval: '1m', '5m', '15m', '30m', '1h'
    period: '7d', '60d', '59d' 등
    """
    try:
        # yfinance로 데이터 다운로드 (prepost=True로 15:30 데이터 확보)
        data = yf.download(tickers=ticker, period=period, interval=interval, prepost=True)
        
        # MultiIndex 컬럼 평탄화 (ValueError 해결)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            st.error("yfinance에서 데이터를 가져오지 못했습니다.")
            return None
        
        # timezone 통일 (UTC로 변환 후 tz-naive로 변경)
        if data.index.tz is not None:
            data.index = data.index.tz_convert('UTC')
        else:
            data.index = data.index.tz_localize('UTC')
        
        # tz-naive로 변환 (한국 시간으로)
        data.index = data.index.tz_convert(None)
        data.index = data.index + pd.Timedelta(hours=9)  # UTC → KST
        
        # Smart Filter: 정규장 시간대 데이터만 필터링 (09:00 ~ 15:30)
        data = data.between_time('09:00', '15:30')
        
        return data
        
    except Exception as e:
        st.error(f"데이터를 불러오지 못했습니다: {str(e)}")
        return None

def get_stock_data(stock_code, start_date, end_date, chart_type='daily'):
    """
    종목 코드로 주식 데이터를 가져오는 함수
    chart_type: 'daily' (일봉), 'minute' (1분봉)
    """
    try:
        if chart_type == 'minute':
            # 1분봉 데이터 가져오기 (최근 30일로 제한)
            stock_data = fdr.DataReader(stock_code, start_date, end_date)
            # 1분봉은 FinanceDataReader에서 지원하지 않으므로 일봉으로 대체
            # 실제 1분봉은 다른 데이터 소스가 필요하지만, 여기서는 일봉을 더 세밀하게 표시
            return stock_data
        else:
            # 일봉 데이터 가져오기
            stock_data = fdr.DataReader(stock_code, start_date, end_date)
            return stock_data
    except Exception as e:
        st.error(f"데이터를 불러오지 못했습니다: {str(e)}")
        return None

def ask_gemini_for_review(df, buy_points, sell_points, buy_avg_price, sell_avg_price, total_buy_quantity, total_sell_quantity, interval_name, api_key):
    """
    Gemini Pro API를 사용한 분할 매매 AI 타점 복기 분석
    """
    try:
        # API Key 유효성 검사
        if not api_key or api_key == "":
            return {
                'ai_analysis': "❌ Gemini API 키가 필요합니다. 사이드바에서 API 키를 입력해주세요.",
                'learning_tips': "API 키를 입력한 후 다시 시도해주세요."
            }
        
        # API Key 설정
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 분할 매매 내역 문자열 생성
        buy_details = []
        for i, point in enumerate(buy_points, 1):
            buy_details.append(f"{i}. {point['date'].strftime('%H:%M')} - {point['price']:,}원 ({point['quantity']:,}주)")
        
        sell_details = []
        for i, point in enumerate(sell_points, 1):
            sell_details.append(f"{i}. {point['date'].strftime('%H:%M')} - {point['price']:,}원 ({point['quantity']:,}주)")
        
        # 분할 매매 AI 프롬프트
        prompt = f"""
        당신은 주식 스캘핑 분석 전문가입니다. 아래 분할 매매 내역을 분석하여 전문적인 조언을 제공해주세요.

        **분할 매매 정보:**
        - 매수 평단가: {buy_avg_price:,.0f}원 (총 {total_buy_quantity:,}주)
        - 매도 평단가: {sell_avg_price:,.0f}원 (총 {total_sell_quantity:,}주)
        - 종합 수익률: {((sell_avg_price - buy_avg_price) / buy_avg_price * 100):+.2f}%
        
        **매수 내역:**
        {chr(10).join(buy_details)}
        
        **매도 내역:**
        {chr(10).join(sell_details)}
        
        **차트 데이터 ({interval_name}):**
        - 시간: {df.index[0].strftime('%H:%M')} ~ {df.index[-1].strftime('%H:%M')}
        - 가격 범위: {df['Low'].min():,.0f}원 ~ {df['High'].max():,.0f}원
        - 최종 종가: {df['Close'].iloc[-1]:,.0f}원
        
        **분석 요청:**
        1. 분할 매매 전략의 장단점 분석
        2. 각 매수/매도 타이밍의 적절성 평가
        3. 평단가와 수익률에 대한 전문적 의견
        4. 향후 스캘핑 매매를 위한 구체적인 개선 제안
        
        반드시 실전적인 조언을 제공해주세요.
        """
        
        # Gemini API 호출
        response = model.generate_content(prompt)
        
        if response and response.text:
            ai_analysis = response.text
            learning_tips = "분할 매매의 평단가 관리와 타이밍 조절이 핵심입니다. 각 매매의 시점과 가격을 꾸준히 기록하고 분석하여 스캘핑 실력을 향상시키세요."
            
            return {
                'ai_analysis': ai_analysis,
                'learning_tips': learning_tips
            }
        else:
            return {
                'ai_analysis': "AI 분석에 실패했습니다. 다시 시도해주세요.",
                'learning_tips': "API 연결 상태를 확인하고 다시 시도해주세요."
            }
            
    except Exception as e:
        return {
            'ai_analysis': f"AI 분석 중 오류가 발생했습니다: {str(e)}",
            'learning_tips': "오류가 지속되면 관리자에게 문의해주세요."
        }
        
        # Gemini API 호출
        response = model.generate_content(expert_prompt)
        ai_analysis = response.text
        
        # 학습 팁 추출 (강화된 파싱 로직)
        learning_tips = ""
        
        # 다양한 형식의 학습 팁 헤딩 처리
        tip_headers = ["## 맞춤형 학습 팁", "## 학습 팁", "**맞춤형 학습 팁**", "**학습 팁**"]
        
        for header in tip_headers:
            if header in ai_analysis:
                try:
                    parts = ai_analysis.split(header)
                    if len(parts) > 1:
                        learning_tips = parts[1].strip()
                        
                        # 다음 섹션이 있다면 잘라내기
                        next_headers = ["## ", "### ", "#### ", "##### ", "###### "]
                        for next_header in next_headers:
                            if next_header in learning_tips and next_header != "## ":
                                learning_tips = learning_tips.split(next_header)[0].strip()
                        
                        # 불필요한 공백 및 개행 정리
                        learning_tips = learning_tips.replace('\n\n', '\n').strip()
                        
                        # 내용이 비어있지 않으면 루프 탈출
                        if learning_tips and len(learning_tips) > 10:
                            break
                except Exception as e:
                    continue
        
        # 학습 팁이 추출되지 않았을 경우의 대비책
        if not learning_tips or len(learning_tips) < 10:
            # AI 답변에서 마지막 문단을 학습 팁으로 추정
            try:
                sentences = ai_analysis.split('\n')
                for sentence in reversed(sentences):
                    if any(keyword in sentence for keyword in ['팁', '학습', '보완', '개선', '기술', '심리']):
                        learning_tips = sentence.strip()
                        break
            except:
                pass
        
        # 최종적으로도 없으면 기본 메시지
        if not learning_tips or len(learning_tips) < 10:
            learning_tips = "이 매매를 통해 스캘핑 타이밍 감각을 더 발전시켜보세요. 꾸준한 복기가 실력 향상의 열쇠입니다."
        
        return {
            'ai_analysis': ai_analysis,
            'learning_tips': learning_tips
        }
        
    except Exception as e:
        return {
            'ai_analysis': f"❌ AI 분석 중 오류 발생: {str(e)}\\n\\nAPI 연결 상태를 확인해주세요.",
            'learning_tips': "API 연결 상태를 확인한 후 다시 시도해주세요."
        }

def calculate_technical_indicators(df):
    """
    필수 보조지표 계산 (RSI, 볼린저 밴드, 이격도, 거래량 증가율)
    """
    try:
        # RSI(14) 계산
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드 계산 (20일 기준, 2표준편차)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # 이격도 계산 (주가와 20분선과의 간격 %)
        if 'MA_20' in df.columns:
            df['Price_MA20_Deviation'] = ((df['Close'] - df['MA_20']) / df['MA_20']) * 100
        
        # 전봉 대비 거래량 증가율 계산
        df['Volume_Ratio'] = (df['Volume'] / df['Volume'].rolling(window=20).mean()) * 100
        
        return df
    except Exception as e:
        st.error(f"보조지표 계산 오류: {str(e)}")
        return df

def get_technical_advice(df, buy_point, sell_point, interval_name):
    """
    스캘핑/단타 전문가 수준의 정밀한 타점 분석
    """
    buy_analysis = []
    sell_analysis = []
    overall_evaluation = []
    
    try:
        # 보조지표 계산
        df = calculate_technical_indicators(df)
        
        # 매수 타점 분석
        buy_price = buy_point['price']
        buy_datetime = pd.to_datetime(buy_point['date'])
        
        # 매수 시점 근처 데이터 찾기
        buy_time_diff = abs(df.index - buy_datetime)
        buy_idx = buy_time_diff.argmin()
        buy_data = df.iloc[buy_idx]
        
        # [매수 타점 분석] - 눌림목 vs 추격매수
        if 'MA_20' in df.columns and not pd.isna(buy_data['MA_20']):
            deviation = buy_data.get('Price_MA20_Deviation', 0)
            if deviation < -2:  # 20분선보다 2% 이하로 눌림
                buy_analysis.append(f"🎯 **눌림목 매수**: 20분선보다 {abs(deviation):.1f}% 하단에서 지지받으며 진입. 전형적인 스캘핑 진입 패턴입니다.")
            elif deviation > 3:  # 20분선보다 3% 이상으로 추격
                buy_analysis.append(f"🚀 **추격매수**: 20분선보다 {deviation:.1f}% 상단에서 급하게 진입. 강세장 스캘핑 전략입니다.")
        
        # [매수 타점 분석] - 볼린저 밴드 분석
        if not pd.isna(buy_data['BB_Upper']) and not pd.isna(buy_data['BB_Lower']):
            bb_position = (buy_data['Close'] - buy_data['BB_Lower']) / (buy_data['BB_Upper'] - buy_data['BB_Lower']) * 100
            if bb_position > 90:
                buy_analysis.append("📈 **볼린저 상단 돌파**: 밴드 상단을 뚫고 급등 진입. 단기 모멘텀 매수 전략입니다.")
            elif bb_position < 10:
                buy_analysis.append("📉 **볼린저 하단 지지**: 밴드 하단에서 지지받으며 진입. 과매도 구간 반등 매수입니다.")
        
        # [매수 타점 분석] - 리스크 평가 (RSI)
        if not pd.isna(buy_data['RSI']):
            rsi = buy_data['RSI']
            if rsi >= 70:
                buy_analysis.append(f"⚠️ **과매수 구간 진입**: RSI {rsi:.1f}로 과매수 경고. 손절 전략 필수입니다.")
            elif rsi <= 30:
                buy_analysis.append(f"✅ **과매도 구간 진입**: RSI {rsi:.1f}로 안전한 진입. 반등 확률 높습니다.")
        
        # [매수 타점 분석] - 거래량 체크
        if not pd.isna(buy_data['Volume_Ratio']):
            vol_ratio = buy_data['Volume_Ratio']
            if vol_ratio >= 200:
                buy_analysis.append(f"🔥 **거래량 폭발**: 평균 대비 {vol_ratio:.0f}% 거래량 터짐. 강력한 매수 신호입니다.")
            elif vol_ratio <= 50:
                buy_analysis.append(f"📊 **거래량 부족**: 평균 대비 {vol_ratio:.0f}% 거래량. 주의 필요합니다.")
        
        # 매도 타점 분석
        sell_price = sell_point['price']
        sell_datetime = pd.to_datetime(sell_point['date'])
        
        # 매도 시점 근처 데이터 찾기
        sell_time_diff = abs(df.index - sell_datetime)
        sell_idx = sell_time_diff.argmin()
        sell_data = df.iloc[sell_idx]
        
        # [매도 타점 분석] - 저항선 매도 여부
        if not pd.isna(sell_data['BB_Upper']):
            if sell_price >= sell_data['BB_Upper'] * 0.98:
                sell_analysis.append("🎯 **볼린저 상단 익절**: 밴드 상단 근처에서 영리하게 익절. 전문가 수준의 타이밍입니다.")
        
        # [매도 타점 분석] - 기회비용 (매도 후 15분 추세)
        if sell_idx + 15 < len(df):
            future_prices = df.iloc[sell_idx+1:sell_idx+16]['Close']
            max_future_price = future_prices.max()
            min_future_price = future_prices.min()
            
            if max_future_price > sell_price * 1.02:  # 2% 이상 추가 상승
                sell_analysis.append(f"📈 **조금 더 길게 볼 수 있었습니다**: 매도 후 15분 내 최고 {((max_future_price/sell_price-1)*100):.1f}% 추가 상승.")
            elif min_future_price < sell_price * 0.98:  # 2% 이상 하락
                sell_analysis.append(f"✅ **완벽한 손절**: 매도 후 15분 내 최대 {((1-min_future_price/sell_price)*100):.1f}% 추가 하락 회피.")
            else:
                sell_analysis.append("✅ **좋은 익절/손절**: 매도 후 15분 내 안정적인 움직임. 적절한 타이밍입니다.")
        
        # 종합 평가 (평단가 기반)
        profit_rate = ((sell_avg_price - buy_avg_price) / buy_avg_price) * 100
        
        # 수익률 기반 평가
        if profit_rate >= 3:
            overall_evaluation.append(f"🏆 **우수한 성과**: {interval_name} 기준 {profit_rate:.1f}% 수익률은 전문가 수준입니다.")
        elif profit_rate >= 1:
            overall_evaluation.append(f"👍 **안정적인 성과**: {interval_name} 기준 {profit_rate:.1f}% 수익률은 스캘핑에 적합합니다.")
        elif profit_rate <= -2:
            overall_evaluation.append(f"⚠️ **개선 필요**: {interval_name} 기준 {abs(profit_rate):.1f}% 손실은 전략 재검토 필요합니다.")
        
        # 리스크 관리 평가
        buy_rsi = buy_data.get('RSI', 50)
        if buy_rsi >= 70 and profit_rate < 0:
            overall_evaluation.append("📉 **리스크 관리 실패**: 과매수 구간 진입 후 손실. 진입 타이밍 개선 필요합니다.")
        elif buy_rsi <= 30 and profit_rate > 1:
            overall_evaluation.append("📈 **리스크 관리 성공**: 과매도 구간 진입 후 수익. 전형적인 스캘핑 성공 사례입니다.")
        
    except Exception as e:
        overall_evaluation.append(f"❌ **분석 오류**: {str(e)}")
    
    return {
        'buy_analysis': buy_analysis,
        'sell_analysis': sell_analysis,
        'overall_evaluation': overall_evaluation
    }

def calculate_moving_averages(df, periods=[5, 10, 20, 60, 120]):
    """
    이동평균선을 계산하는 함수
    """
    for period in periods:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
    return df

def create_candlestick_chart(df, show_ma_5=True, show_ma_10=True, show_ma_20=True, 
                           show_ma_60=True, show_ma_120=True, buy_points=None, sell_points=None, chart_type='daily'):
    """
    HTS 스타일의 캔들 차트 생성 함수
    """
    if df.empty:
        return None
    
    # 서브플롯 생성 (캔들 + 거래량)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.8, 0.2]
    )
    
    # 캔들 차트 추가 (X축을 카테고리로 변경하여 빈 공간 제거)
    time_strings = df.index.strftime('%H:%M').tolist()  # 시간을 문자열로 변환
    
    fig.add_trace(
        go.Candlestick(
            x=time_strings,  # Datetime 대신 문자열 사용
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='캔들',
            increasing_line_color='red',
            decreasing_line_color='blue',
            increasing_fillcolor='red',
            decreasing_fillcolor='blue',
            line=dict(width=1)  # 날렵한 테두리
        ),
        row=1, col=1
    )
    
    # 이동평균선 추가 (고대비 색상 적용)
    ma_colors = {
        'MA_5': '#FF0000',    # 빨강 (MA5)
        'MA_10': '#00B050',   # 초록 (MA10)
        'MA_20': '#FFC000',   # 노랑 (MA20)
        'MA_60': '#0070C0',   # 파랑 (MA60)
        'MA_120': '#7030A0'   # 보라 (MA120)
    }
    
    if show_ma_5 and 'MA_5' in df.columns:
        fig.add_trace(
            go.Scatter(x=time_strings, y=df['MA_5'], name='MA5', line=dict(color=ma_colors['MA_5'], width=1.5)),
            row=1, col=1
        )
    
    if show_ma_10 and 'MA_10' in df.columns:
        fig.add_trace(
            go.Scatter(x=time_strings, y=df['MA_10'], name='MA10', line=dict(color=ma_colors['MA_10'], width=1.5)),
            row=1, col=1
        )
    
    if show_ma_20 and 'MA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=time_strings, y=df['MA_20'], name='MA20', line=dict(color=ma_colors['MA_20'], width=1.5)),
            row=1, col=1
        )
    
    if show_ma_60 and 'MA_60' in df.columns:
        fig.add_trace(
            go.Scatter(x=time_strings, y=df['MA_60'], name='MA60', line=dict(color=ma_colors['MA_60'], width=1.5)),
            row=1, col=1
        )
    
    if show_ma_120 and 'MA_120' in df.columns:
        fig.add_trace(
            go.Scatter(x=time_strings, y=df['MA_120'], name='MA120', line=dict(color=ma_colors['MA_120'], width=1.5)),
            row=1, col=1
        )
    
    # 분할 매수/매도 포인트 MTS 꼬리표 마커 추가 (다중 매매 정보 상세 호버 텍스트)
    if buy_points:
        # 매수 포인트를 캔들 시간으로 그룹화하여 다중 매매 정보 처리
        buy_grouped = {}
        
        for buy_point in buy_points:
            buy_datetime = pd.to_datetime(buy_point['date'])
            if buy_datetime.tz is not None:
                buy_datetime = buy_datetime.tz_convert(None)
            
            # 시간 매핑: 해당 매매 시간이 속한 캔들의 시간으로 매핑
            try:
                # df.index.asof()로 가장 가까운 이전 캔들 시간 찾기
                mapped_time = df.index.asof(buy_datetime)
                if mapped_time is None:
                    continue  # 해당 시간에 데이터가 없으면 스킵
                
                # 매핑된 시간을 문자열로 변환
                buy_time_str = mapped_time.strftime('%H:%M')
                
                # 그룹화된 데이터에 추가
                if buy_time_str not in buy_grouped:
                    buy_grouped[buy_time_str] = {
                        'mapped_time': mapped_time,
                        'trades': []
                    }
                
                buy_grouped[buy_time_str]['trades'].append({
                    'price': buy_point['price'],
                    'quantity': buy_point['quantity']
                })
                
            except Exception as e:
                st.warning(f"매수 시간 매핑 오류: {buy_datetime.strftime('%H:%M')} - {str(e)}")
                continue
        
        # 그룹화된 매수 데이터로 마커 생성
        for time_str, group_data in buy_grouped.items():
            try:
                mapped_time = group_data['mapped_time']
                trades = group_data['trades']
                
                # 매핑된 시간의 캔들 데이터 찾기
                candle_data = df[df.index == mapped_time]
                if candle_data.empty:
                    continue
                
                candle = candle_data.iloc[0]
                
                # 평단가 계산
                total_amount = sum(trade['price'] * trade['quantity'] for trade in trades)
                total_quantity = sum(trade['quantity'] for trade in trades)
                avg_price = total_amount / total_quantity if total_quantity > 0 else 0
                
                # 캔들 저가보다 살짝 아래에 위치 (평단가 기준)
                marker_y = candle['Low'] * 0.995
                
                # 상세 호버 텍스트 생성 (HTML 형식)
                hover_lines = [f"<b>[다중 매수 내역]</b>"]
                for i, trade in enumerate(trades, 1):
                    hover_lines.append(f"매수 {i}: {trade['price']:,}원 ({trade['quantity']:,}주)")
                hover_lines.append("---")
                hover_lines.append(f"<b>총 수량: {total_quantity:,}주 / 평단: {avg_price:,.0f}원</b>")
                
                hovertext = "<br>".join(hover_lines)
                
                # MTS 스타일 매수 꼬리표 (아래에서 위로 찌름)
                fig.add_annotation(
                    x=time_str,  # 매핑된 문자열 시간 사용
                    y=marker_y,
                    text='B',
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1.5,
                    arrowwidth=2,
                    ay=30,  # 아래에서 위로 찌름
                    ax=0,
                    bgcolor='#FFB800',  # 노란색 배경
                    font=dict(color='black', size=10, family="Arial Black"),
                    bordercolor='#FFB800',
                    borderwidth=2,
                    hovertext=hovertext,  # 상세 다중 매매 정보
                    hoverlabel=dict(bgcolor='#FFB800', font_color='black')
                )
            except Exception as e:
                st.warning(f"매수 마커 생성 오류: {time_str} - {str(e)}")
                continue
    
    if sell_points:
        # 매도 포인트를 캔들 시간으로 그룹화하여 다중 매매 정보 처리
        sell_grouped = {}
        
        for sell_point in sell_points:
            sell_datetime = pd.to_datetime(sell_point['date'])
            if sell_datetime.tz is not None:
                sell_datetime = sell_datetime.tz_convert(None)
            
            # 시간 매핑: 해당 매매 시간이 속한 캔들의 시간으로 매핑
            try:
                # df.index.asof()로 가장 가까운 이전 캔들 시간 찾기
                mapped_time = df.index.asof(sell_datetime)
                if mapped_time is None:
                    continue  # 해당 시간에 데이터가 없으면 스킵
                
                # 매핑된 시간을 문자열로 변환
                sell_time_str = mapped_time.strftime('%H:%M')
                
                # 그룹화된 데이터에 추가
                if sell_time_str not in sell_grouped:
                    sell_grouped[sell_time_str] = {
                        'mapped_time': mapped_time,
                        'trades': []
                    }
                
                sell_grouped[sell_time_str]['trades'].append({
                    'price': sell_point['price'],
                    'quantity': sell_point['quantity']
                })
                
            except Exception as e:
                st.warning(f"매도 시간 매핑 오류: {sell_datetime.strftime('%H:%M')} - {str(e)}")
                continue
        
        # 그룹화된 매도 데이터로 마커 생성
        for time_str, group_data in sell_grouped.items():
            try:
                mapped_time = group_data['mapped_time']
                trades = group_data['trades']
                
                # 매핑된 시간의 캔들 데이터 찾기
                candle_data = df[df.index == mapped_time]
                if candle_data.empty:
                    continue
                
                candle = candle_data.iloc[0]
                
                # 평단가 계산
                total_amount = sum(trade['price'] * trade['quantity'] for trade in trades)
                total_quantity = sum(trade['quantity'] for trade in trades)
                avg_price = total_amount / total_quantity if total_quantity > 0 else 0
                
                # 캔들 고가보다 살짝 위에 위치 (평단가 기준)
                marker_y = candle['High'] * 1.005
                
                # 상세 호버 텍스트 생성 (HTML 형식)
                hover_lines = [f"<b>[다중 매도 내역]</b>"]
                for i, trade in enumerate(trades, 1):
                    hover_lines.append(f"매도 {i}: {trade['price']:,}원 ({trade['quantity']:,}주)")
                hover_lines.append("---")
                hover_lines.append(f"<b>총 수량: {total_quantity:,}주 / 평단: {avg_price:,.0f}원</b>")
                
                hovertext = "<br>".join(hover_lines)
                
                # MTS 스타일 매도 꼬리표 (위에서 아래로 찌름)
                fig.add_annotation(
                    x=time_str,  # 매핑된 문자열 시간 사용
                    y=marker_y,
                    text='S',
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1.5,
                    arrowwidth=2,
                    ay=-30,  # 위에서 아래로 찌름
                    ax=0,
                    bgcolor='#00B0F0',  # 하늘색 배경
                    font=dict(color='white', size=10, family="Arial Black"),
                    bordercolor='#00B0F0',
                    borderwidth=2,
                    hovertext=hovertext,  # 상세 다중 매매 정보
                    hoverlabel=dict(bgcolor='#00B0F0', font_color='white')
                )
            except Exception as e:
                st.warning(f"매도 마커 생성 오류: {time_str} - {str(e)}")
                continue
    
    # 거래량 차트 추가 (전봉 종가 기준 색상)
    colors = []
    for i in range(len(df)):
        if i == 0:
            # 첫 데이터는 시가와 종가 비교
            if df['Close'].iloc[i] >= df['Open'].iloc[i]:
                colors.append('#FF6B6B')  # 양봉 (빨강)
            else:
                colors.append('#4ECDC4')  # 음봉 (청록)
        else:
            # 전봉 종가 대비 현재 종가 비교
            if df['Close'].iloc[i] >= df['Close'].iloc[i-1]:
                colors.append('#FF6B6B')  # 상승 (빨강)
            else:
                colors.append('#4ECDC4')  # 하락 (청록)
    
    fig.add_trace(
        go.Bar(
            x=time_strings,  # 문자열 시간 사용
            y=df['Volume'], 
            name='거래량', 
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # HTS 스타일 레이아웃 설정 (Drag to Pan 포함)
    fig.update_layout(
        title=dict(
            text=f'주식 분봉 차트 ({chart_type.upper() if chart_type != "intraday" else "분봉"})',
            x=0.5,  # 중앙 정렬
            font=dict(size=16, color='black')
        ),
        yaxis_title='가격 (원)',
        xaxis_title='시간',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='white',
        
        # HTS 스타일 십자선 완벽 설정
        hovermode='x unified',  # 통합 호버 모드 강제
        font=dict(color='black'),  # 글씨 색상 명시적 설정
        dragmode='pan',  # HTS 스타일 드래그 (박스 확대 대신 좌우 이동)
        
        # X축 십자선 설정 (확대/축소 가능)
        xaxis=dict(
            showgrid=True,
            gridcolor='#E0E0E0',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='#CCCCCC',
            zerolinewidth=1,
            showspikes=True,           # 십자선 표시
            spikemode='across',        # 가로로 뻗는 십자선
            spikesnap='cursor',        # 커서 따라 움직임
            showline=True,
            spikedash='solid',         # 실선
            spikecolor='gray',         # 회색 십자선
            spikethickness=1,          # 십자선 두께
            fixedrange=False           # 확대/축소 가능
        ),
        
        # Y축 십자선 설정 (실시간 가격 레이블 포함)
        yaxis=dict(
            showgrid=True,
            gridcolor='#E0E0E0',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='#CCCCCC',
            zerolinewidth=1,
            showspikes=True,           # 십자선 표시
            spikemode='across',        # 세로로 뻗는 십자선
            spikesnap='cursor',        # 커서 따라 움직임
            showline=True,
            spikedash='solid',         # 실선
            spikecolor='gray',         # 회색 십자선
            spikethickness=1,          # 십자선 두께
            hoverformat='.0f',         # 실시간 가격 표시
            fixedrange=False           # 확대/축소 가능
        )
    )
    
    # 분봉용 X축 포맷팅 및 카테고리 설정
    if chart_type == 'intraday':
        fig.update_xaxes(
            type='category', 
            categoryorder='category ascending',  # 시간 순서대로 정렬
            tickformat='%H:%M'
        )
    
    # 정적 이평선 텍스트 박스 삭제 (불필요한 UI 제거)
    
    fig.update_yaxes(title_text="가격 (원)", row=1, col=1, gridcolor='#E0E0E0')
    fig.update_yaxes(title_text="거래량", row=2, col=1, gridcolor='#E0E0E0')
    fig.update_xaxes(gridcolor='#E0E0E0')
    
    return fig

# 메뉴 1: 내 매매 타점 복기
if menu == "내 매매 타점 복기":
    st.header("📊 내 매매 타점 복기")
    
    # Gemini API 키 입력 (사이드바 최상단)
    st.bar.subheader("🔑 API 설정")
    
    # 1. 초기 키 값 설정 (st.secrets에서 자동 불러오기)
    default_api_key = ""
    if "GEMINI_API_KEY" in st.secrets:
        default_api_key = st.secrets["GEMINI_API_KEY"]
    
    # 2. 텍스트 인풋의 value에 기본값 할당
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key", 
        type="password", 
        value=default_api_key,
        help=".streamlit/secrets.toml 파일에 GEMINI_API_KEY를 저장하면 자동으로 불러옵니다."
    )
    
    # 사이드바에 입력 폼 생성
    st.sidebar.subheader("매매 정보 입력")
    
    # 종목명 입력 (자동완성 기능)
    stock_name_input = st.sidebar.text_input("종목명 (예: 삼성전자)", "삼성전자")
    
    # 종목명 자동완성 드롭다운
    if stock_name_input and len(stock_name_input) >= 1:
        suggestions = get_stock_suggestions(stock_name_input)
        if suggestions:
            selected_stock = st.sidebar.selectbox(
                "종목 선택 (자동완성):",
                options=suggestions,
                index=0 if suggestions else None,
                help="입력한 글자가 포함된 종목들입니다."
            )
            stock_name = selected_stock
        else:
            stock_name = stock_name_input
    else:
        stock_name = stock_name_input
    
    # 분할 매수/매도 입력 UI (실전 스캘핑 핵심 기능)
    st.sidebar.subheader("📊 분할 매매 내역 입력")
    
    # 매수 내역 표 (모바일 최적화 UI)
    st.sidebar.write("**매수 내역**")
    default_buy_data = pd.DataFrame({
        '날짜': [datetime.datetime.now().date()],
        '시간': ['09:30'],  # 문자열로 시간 저장
        '가격': [75000],
        '수량': [100]
    })
    
    buy_data = st.sidebar.data_editor(
        default_buy_data,
        num_rows="dynamic",
        use_container_width=True,  # 모바일 호환성
        column_config={
            "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
            "시간": st.column_config.TextColumn(
                "시간 (24시간제)", 
                help="24시간제로 입력해주세요 (예: 09:30, 14:15, 15:30)"
            ),
            "가격": st.column_config.NumberColumn("가격", format="%,d원"),
            "수량": st.column_config.NumberColumn("수량", format="%,d주")
        },
        key="buy_editor"
    )
    
    # 매도 내역 표 (모바일 최적화 UI)
    st.sidebar.write("**매도 내역**")
    default_sell_data = pd.DataFrame({
        '날짜': [datetime.datetime.now().date()],
        '시간': ['10:00'],  # 문자열로 시간 저장
        '가격': [80000],
        '수량': [100]
    })
    
    sell_data = st.sidebar.data_editor(
        default_sell_data,
        num_rows="dynamic",
        use_container_width=True,  # 모바일 호환성
        column_config={
            "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
            "시간": st.column_config.TextColumn(
                "시간 (24시간제)", 
                help="24시간제로 입력해주세요 (예: 09:30, 14:15, 15:30)"
            ),
            "가격": st.column_config.NumberColumn("가격", format="%,d원"),
            "수량": st.column_config.NumberColumn("수량", format="%,d주")
        },
        key="sell_editor"
    )
    
    # 분봉 간격 선택 (스캘핑을 위한 핵심 기능)
    st.sidebar.subheader("📊 분봉 간격 선택")
    interval_options = {
        "1분봉": {"resample": None, "period": "7d"},
        "3분봉": {"resample": "3T", "period": "7d"},
        "5분봉": {"resample": "5T", "period": "60d"},
        "10분봉": {"resample": "10T", "period": "60d"}
    }
    
    selected_interval_name = st.sidebar.radio(
        "분봉 간격:",
        options=list(interval_options.keys()),
        index=0,
        help="스캘핑(초단타) 매매를 위한 분봉 데이터를 선택합니다."
    )
    
    interval_config = interval_options[selected_interval_name]
    chart_type = 'intraday'  # 무조건 분봉 모드 사용
    
    st.sidebar.info(f"📊 {selected_interval_name} 차트를 표시합니다 (yfinance 기반)")
    
    # 이동평균선 체크박스
    st.sidebar.subheader("이동평균선 표시")
    show_ma_5 = st.sidebar.checkbox("5일 이동평균선", value=True)
    show_ma_10 = st.sidebar.checkbox("10일 이동평균선", value=True)
    show_ma_20 = st.sidebar.checkbox("20일 이동평균선", value=True)
    show_ma_60 = st.sidebar.checkbox("60일 이동평균선", value=True)
    show_ma_120 = st.sidebar.checkbox("120일 이동평균선", value=True)
    
    # 데이터 조회 버튼
    if st.sidebar.button("차트 보기"):
        if stock_name:
            # 분할 매매 데이터 처리
            buy_points = []
            sell_points = []
            
            # 매수 내역 처리
            for _, row in buy_data.iterrows():
                if pd.notna(row['날짜']) and pd.notna(row['시간']) and pd.notna(row['가격']) and pd.notna(row['수량']):
                    # 날짜와 시간을 결합하여 datetime 생성
                    buy_datetime = pd.to_datetime(f"{row['날짜']} {row['시간']}")
                    buy_points.append({
                        'date': buy_datetime,
                        'price': row['가격'],
                        'quantity': row['수량']
                    })
            
            # 매도 내역 처리
            for _, row in sell_data.iterrows():
                if pd.notna(row['날짜']) and pd.notna(row['시간']) and pd.notna(row['가격']) and pd.notna(row['수량']):
                    # 날짜와 시간을 결합하여 datetime 생성
                    sell_datetime = pd.to_datetime(f"{row['날짜']} {row['시간']}")
                    sell_points.append({
                        'date': sell_datetime,
                        'price': row['가격'],
                        'quantity': row['수량']
                    })
            
            if not buy_points or not sell_points:
                st.error("매수 또는 매도 내역을 입력해주세요.")
                st.stop()
            
            # 평단가 계산
            total_buy_amount = sum(p['price'] * p['quantity'] for p in buy_points)
            total_buy_quantity = sum(p['quantity'] for p in buy_points)
            buy_avg_price = total_buy_amount / total_buy_quantity if total_buy_quantity > 0 else 0
            
            total_sell_amount = sum(p['price'] * p['quantity'] for p in sell_points)
            total_sell_quantity = sum(p['quantity'] for p in sell_points)
            sell_avg_price = total_sell_amount / total_sell_quantity if total_sell_quantity > 0 else 0
            
            # 종합 수익률 계산
            profit_rate = ((sell_avg_price - buy_avg_price) / buy_avg_price) * 100 if buy_avg_price > 0 else 0
            
            # 평단가 정보 표시
            st.info(f"📊 **평단가 정보**: 총 매수 평단가: {buy_avg_price:,.0f}원 / 총 매도 평단가: {sell_avg_price:,.0f}원 / 종합 수익률: {profit_rate:+.2f}%")
            
            # yfinance 티커로 변환
            yf_ticker = get_yfinance_ticker(stock_name)
            
            if yf_ticker:
                st.write(f"🔍 종목명 '{stock_name}' → yfinance 티커 '{yf_ticker}' 변환 완료")
                
                # yfinance로 1분봉 데이터 가져오기 (무조건 1분봉으로 7일치 로드)
                stock_data = get_intraday_data(
                    ticker=yf_ticker,
                    interval='1m',  # 무조건 1분봉으로 로드
                    period='7d'    # 7일치 데이터
                )
                
                # 데이터 유효성 검사
                if stock_data is None or stock_data.empty:
                    st.error("분봉 데이터를 불러오지 못했습니다. 티커나 인터넷 연결을 확인해주세요.")
                    st.stop()
                
                # Pandas Resampling 적용 (사용자 선택 분봉으로 변환)
                resample_rule = interval_config["resample"]
                if resample_rule is not None:
                    stock_data = resample_data(stock_data, resample_rule)
                    st.info(f"📊 {selected_interval_name}로 리샘플링 완료 ({len(stock_data)}개 봉)")
                
                # **강제 순서 1**: 리샘플링된 데이터로 이동평균선 계산
                stock_data = calculate_moving_averages(stock_data, periods=[5, 10, 20, 60, 120])
                
                # **강제 순서 2**: 전체 데이터 상태에서 보조지표 계산
                stock_data = calculate_technical_indicators(stock_data)
                
                # **강제 순서 3**: 모든 계산 끝난 후에만 당일 데이터 필터링
                if buy_points and sell_points:
                    target_date = buy_points[0]['date'].date()
                    
                    # 보조지표 계산된 상태에서 당일 데이터만 필터링
                    stock_data = stock_data[stock_data.index.date == pd.Timestamp(target_date).date()]
                    
                    if stock_data.empty:
                        st.error(f"{target_date}의 분봉 데이터가 없습니다.")
                        st.stop()
                
                # 분봉 차트 생성 (분할 매매 데이터 전달)
                fig = create_candlestick_chart(
                    stock_data, 
                    show_ma_5, show_ma_10, show_ma_20, show_ma_60, show_ma_120,  # 모든 이평선 온전히 전달
                    buy_points, sell_points, chart_type
                )
                
                if fig:
                    # 차트 표시 (스크롤 확대/축소 활성화)
                    st.plotly_chart(fig, use_container_width=True, theme=None, config={'scrollZoom': True})
                    
                    # 진짜 AI 복기 분석 (Gemini Pro)
                    st.subheader("🤖 진짜 AI 타점 복기 분석")
                    
                    # API 키 확인
                    if not gemini_api_key:
                        st.error("🔑 Gemini API 키가 필요합니다. 사이드바에서 API 키를 입력해주세요.")
                        st.stop()
                    
                    with st.spinner("🧠 AI가 분할 매매를 분석 중입니다..."):
                        ai_result = ask_gemini_for_review(
                            stock_data, 
                            buy_points, 
                            sell_points, 
                            buy_avg_price,
                            sell_avg_price,
                            total_buy_quantity,
                            total_sell_quantity,
                            selected_interval_name,
                            gemini_api_key  # API 키 전달
                        )
                    
                    # AI 분석 결과 표시
                    st.markdown("---")
                    st.markdown("### 📊 AI 전문가 분석 결과")
                    st.markdown(ai_result['ai_analysis'])
                    
                    # 맞춤형 학습 팁 (눈에 띄는 UI)
                    if ai_result['learning_tips']:
                        st.markdown("---")
                        st.success("### 🎓 오늘의 매매 교훈")
                        st.info(ai_result['learning_tips'])
                    
                    # 수익률 계산 (평단가 변수 사용)
                    profit_rate = ((sell_avg_price - buy_avg_price) / buy_avg_price) * 100
                    st.markdown("---")
                    st.success(f"## 💰 매매 수익률: {profit_rate:.2f}%")
                
            else:
                st.error(f"'{stock_name}' 종목을 yfinance 티커로 변환할 수 없습니다. 정확한 종목명을 입력해주세요.")
        else:
            st.warning("종목명을 입력해주세요.")

# 메뉴 2: 차트 트레이닝 퀴즈
elif menu == "차트 트레이닝 퀴즈":
    st.header("🎯 차트 트레이닝 퀴즈")
    st.write("듀얼링고 스타일의 주식 차트 퀴즈입니다. 과거 차트를 보고 매수/관망을 결정해보세요!")
    
    # 사이드바에 퀴즈 설정
    st.sidebar.subheader("퀴즈 설정")
    
    # 종목명 입력
    quiz_stock_name = st.sidebar.text_input("퀴즈 종목명", "삼성전자")
    
    # 이동평균선 체크박스
    st.sidebar.subheader("이동평균선 표시")
    quiz_show_ma_5 = st.sidebar.checkbox("5일 이동평균선", value=True)
    quiz_show_ma_10 = st.sidebar.checkbox("10일 이동평균선", value=True)
    quiz_show_ma_20 = st.sidebar.checkbox("20일 이동평균선", value=True)
    quiz_show_ma_60 = st.sidebar.checkbox("60일 이동평균선", value=False)
    quiz_show_ma_120 = st.sidebar.checkbox("120일 이동평균선", value=False)
    
    # 퀴즈 시작 버튼
    if st.sidebar.button("새 퀴즈 시작"):
        if quiz_stock_name:
            # 종목 코드 찾기
            stock_code = get_stock_code(quiz_stock_name)
            
            if stock_code:
                # 1년 치 데이터 가져오기
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=365)
                
                stock_data = get_stock_data(stock_code, start_date, end_date)
                
                if stock_data is not None and not stock_data.empty:
                    # 이동평균선 계산
                    stock_data = calculate_moving_averages(stock_data)
                    
                    # 무작위로 기준일 선택 (데이터의 앞 70% 중에서)
                    data_length = len(stock_data)
                    split_point = int(data_length * 0.7)
                    random_index = random.randint(60, split_point)  # 60일 이후부터 선택
                    
                    quiz_date = stock_data.index[random_index]
                    
                    # 기준일까지의 데이터만 보여주기
                    quiz_data = stock_data.iloc[:random_index + 1]
                    
                    # 세션 상태에 퀴즈 정보 저장
                    st.session_state.quiz_data = stock_data
                    st.session_state.quiz_data_visible = quiz_data
                    st.session_state.quiz_date = quiz_date
                    st.session_state.quiz_answered = False
                    st.session_state.show_ma = {
                        'ma_5': quiz_show_ma_5,
                        'ma_10': quiz_show_ma_10,
                        'ma_20': quiz_show_ma_20,
                        'ma_60': quiz_show_ma_60,
                        'ma_120': quiz_show_ma_120
                    }
                    
                    st.success(f"새 퀴즈가 생성되었습니다! {quiz_date.strftime('%Y년 %m월 %d일')}까지의 차트를 보고 결정하세요.")
                    
                else:
                    st.error("주식 데이터를 불러올 수 없습니다.")
            else:
                st.error(f"'{quiz_stock_name}' 종목을 찾을 수 없습니다.")
    
    # 퀴즈가 생성되었을 경우
    if 'quiz_data_visible' in st.session_state and not st.session_state.get('quiz_answered', True):
        st.subheader(f"📈 {st.session_state.quiz_date.strftime('%Y년 %m월 %d일')}까지의 차트")
        
        # 차트 생성 (기준일까지의 데이터만)
        fig = create_candlestick_chart(
            st.session_state.quiz_data_visible,
            st.session_state.show_ma['ma_5'],
            st.session_state.show_ma['ma_10'],
            st.session_state.show_ma['ma_20'],
            st.session_state.show_ma['ma_60'],
            st.session_state.show_ma['ma_120']
        )
        
        # 기준일 이후는 회색으로 표시
        fig.add_vline(
            x=st.session_state.quiz_date, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="퀴즈 기준일"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 선택 버튼
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🟢 매수한다", use_container_width=True):
                st.session_state.user_choice = "매수"
                st.session_state.quiz_answered = True
        
        with col2:
            if st.button("🔵 관망한다", use_container_width=True):
                st.session_state.user_choice = "관망"
                st.session_state.quiz_answered = True
    
    # 정답 공개
    if st.session_state.get('quiz_answered', False):
        st.subheader("✅ 정답 공개")
        
        # 전체 차트 보여주기
        full_fig = create_candlestick_chart(
            st.session_state.quiz_data,
            st.session_state.show_ma['ma_5'],
            st.session_state.show_ma['ma_10'],
            st.session_state.show_ma['ma_20'],
            st.session_state.show_ma['ma_60'],
            st.session_state.show_ma['ma_120']
        )
        
        # 기준일 표시
        full_fig.add_vline(
            x=st.session_state.quiz_date, 
            line_dash="dash", 
            line_color="red",
            annotation_text="퀴즈 기준일"
        )
        
        st.plotly_chart(full_fig, use_container_width=True)
        
        # 사용자 선택 표시
        user_choice = st.session_state.get('user_choice', '')
        st.write(f"**당신의 선택:** {user_choice}")
        
        # 해설
        st.subheader("📝 해설")
        
        # 기준일 이후 1개월 데이터
        quiz_date_index = st.session_state.quiz_data.index.get_loc(st.session_state.quiz_date)
        future_data = st.session_state.quiz_data.iloc[quiz_date_index:quiz_date_index + 20]  # 약 1개월
        
        if len(future_data) > 0:
            start_price = future_data.iloc[0]['Close']
            end_price = future_data.iloc[-1]['Close']
            change_rate = ((end_price - start_price) / start_price) * 100
            
            if change_rate > 5:
                correct_answer = "매수"
                explanation = f"이 자리에서는 약 {change_rate:.1f}% 상승했습니다. 거래량이 증가하며 상승 추세가 이어진 좋은 매수 타이밍이었습니다."
            elif change_rate < -5:
                correct_answer = "관망"
                explanation = f"이 자리에서는 약 {abs(change_rate):.1f}% 하락했습니다. 하락 추세가 예상되어 관망이 옳은 선택이었습니다."
            else:
                correct_answer = "관망"
                explanation = f"이 자리에서는 {change_rate:.1f}%의 횡보장이었습니다. 큰 변동이 없어 관망이 합리적인 선택이었습니다."
            
            st.info(explanation)
            
            # 정답 확인
            if user_choice == correct_answer:
                st.success("🎉 정답입니다! 좋은 판단이셨습니다.")
            else:
                st.warning(f"아쉽네요. 정답은 '{correct_answer}'였습니다.")
        
        # 새 퀴즈 버튼
        if st.button("🔄 새 퀴즈 시작하기"):
            for key in ['quiz_data', 'quiz_data_visible', 'quiz_date', 'quiz_answered', 'user_choice', 'show_ma']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
