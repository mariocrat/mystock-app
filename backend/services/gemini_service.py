import google.generativeai as genai
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def ask_gemini_for_review(
    df: pd.DataFrame, 
    buy_points: list, 
    sell_points: list, 
    buy_avg_price: float, 
    sell_avg_price: float, 
    total_buy_quantity: int, 
    total_sell_quantity: int, 
    interval_name: str
):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            'ai_analysis': "❌ Gemini API 키가 서버에 설정되어 있지 않습니다.",
            'learning_tips': "서버 관리자에게 문의하세요."
        }
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        buy_details = []
        for i, point in enumerate(buy_points, 1):
            buy_details.append(f"{i}. {point['date']} - {point['price']} ({point['quantity']}주)")
            
        sell_details = []
        for i, point in enumerate(sell_points, 1):
            sell_details.append(f"{i}. {point['date']} - {point['price']} ({point['quantity']}주)")
            
        prompt = f"""
        당신은 주식 스캘핑 분석 전문가입니다. 아래 분할 매매 내역을 분석하여 전문적인 조언을 제공해주세요.

        **분할 매매 정보:**
        - 매수 평단가: {buy_avg_price:,.2f} (총 {total_buy_quantity:,}주)
        - 매도 평단가: {sell_avg_price:,.2f} (총 {total_sell_quantity:,}주)
        - 종합 수익률: {((sell_avg_price - buy_avg_price) / buy_avg_price * 100):+.2f}% if buy_avg_price > 0 else 0%
        
        **매수 내역:**
        {chr(10).join(buy_details)}
        
        **매도 내역:**
        {chr(10).join(sell_details)}
        
        **차트 데이터 ({interval_name}):**
        - 시간: {df.index[0]} ~ {df.index[-1]}
        - 가격 범위: {df['Low'].min():,.2f} ~ {df['High'].max():,.2f}
        - 최종 종가: {df['Close'].iloc[-1]:,.2f}
        
        **분석 요청:**
        1. 분할 매매 전략의 장단점 분석
        2. 각 매수/매도 타이밍의 적절성 평가
        3. 평단가와 수익률에 대한 전문적 의견
        4. 향후 스캘핑 매매를 위한 구체적인 개선 제안
        
        반드시 실전적인 조언을 제공해주세요.
        ## 맞춤형 학습 팁
        이 부분 아래에 오늘의 매매 교훈과 학습 팁을 1~2문장으로 요약해서 적어주세요.
        """
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            ai_analysis = response.text
            
            # 파싱 로직
            learning_tips = ""
            if "## 맞춤형 학습 팁" in ai_analysis:
                parts = ai_analysis.split("## 맞춤형 학습 팁")
                learning_tips = parts[1].strip()
                ai_analysis = parts[0].strip()
                
            if not learning_tips:
                learning_tips = "분할 매매의 평단가 관리와 타이밍 조절이 핵심입니다."
                
            return {
                'ai_analysis': ai_analysis,
                'learning_tips': learning_tips
            }
        else:
            return {
                'ai_analysis': "AI 분석에 실패했습니다.",
                'learning_tips': "다시 시도해주세요."
            }
            
    except Exception as e:
        return {
            'ai_analysis': f"AI 분석 중 오류가 발생했습니다: {str(e)}",
            'learning_tips': "오류가 지속되면 관리자에게 문의해주세요."
        }
