"use client";
import React, { useState } from "react";
import MarketSelector from "@/components/MarketSelector";
import TradeInputList, { TradePoint } from "@/components/TradeInputList";
import ChartViewer from "@/components/ChartViewer";
import AdMobBanner from "@/components/AdMobBanner";
import { useMarket } from "@/contexts/MarketContext";

import StockSearch from "@/components/StockSearch";

export default function Home() {
  const { market } = useMarket();
  const [ticker, setTicker] = useState("");
  const [trades, setTrades] = useState<TradePoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);
  const [analysis, setAnalysis] = useState<{ ai_analysis: string, learning_tips: string } | null>(null);

  const analyzeTrades = async () => {
    if (!ticker) return alert(market === "KR" ? "종목명을 검색 후 선택해주세요." : "티커를 입력해주세요.");
    if (trades.length === 0) return alert("매매 내역을 하나 이상 입력해주세요.");

    setLoading(true);
    setAnalysis(null);
    setChartData([]);

    try {
      // 1. 차트 데이터 가져오기
      const dataRes = await fetch("http://localhost:8000/api/market/data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker,
          market,
          interval: "1m",
          period: "7d"
        })
      });

      if (!dataRes.ok) {
        throw new Error("차트 데이터를 불러오는데 실패했습니다.");
      }
      
      const marketData = await dataRes.json();
      setChartData(marketData.data);

      // 2. AI 복기 분석 요청
      const aiRes = await fetch("http://localhost:8000/api/analysis/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker,
          market,
          buy_points: trades.filter(t => t.type === "buy").map(t => ({ date: t.date, price: t.price, quantity: t.quantity })),
          sell_points: trades.filter(t => t.type === "sell").map(t => ({ date: t.date, price: t.price, quantity: t.quantity })),
          interval_name: "1분봉"
        })
      });

      if (!aiRes.ok) {
        const errorData = await aiRes.json();
        throw new Error(errorData.detail || "AI 분석 요청에 실패했습니다.");
      } 
      
      const aiData = await aiRes.json();
      setAnalysis(aiData);

    } catch (err: any) {
      alert(err.message || "서버 통신 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-3xl mx-auto p-4 sm:p-6 w-full flex-1 pb-24">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold bg-gradient-to-r from-indigo-600 to-pink-500 bg-clip-text text-transparent">
          AI 주식 매매 복기
        </h1>
        <p className="text-slate-500 mt-2 font-medium">모바일 환경에 최적화된 스캘핑 타점 분석</p>
      </div>

      <MarketSelector />

      <div className="glass-card p-6 mb-6">
        <h3 className="text-lg font-bold text-slate-800 mb-4">종목 설정</h3>
        <StockSearch ticker={ticker} setTicker={setTicker} />
      </div>

      <TradeInputList trades={trades} setTrades={setTrades} />

      <button 
        className="btn-primary mb-6" 
        onClick={analyzeTrades} 
        disabled={loading}
      >
        {loading ? "데이터 수집 및 AI 분석 중..." : "🚀 AI 타점 복기 시작"}
      </button>

      {chartData.length > 0 && <ChartViewer data={chartData} trades={trades} market={market} />}

      {analysis && (
        <div className="glass-card p-6 mb-6 border-indigo-100">
          <h3 className="text-xl font-bold text-indigo-600 mb-4">🤖 AI 전문가 분석 결과</h3>
          <div className="whitespace-pre-wrap leading-relaxed text-[15px] text-slate-700">
            {analysis.ai_analysis}
          </div>
          
          <div className="mt-6 p-4 bg-indigo-50/50 rounded-xl border-l-4 border-indigo-500">
            <h4 className="text-indigo-700 font-bold mb-2">💡 맞춤형 학습 팁</h4>
            <p className="text-slate-700 font-medium">{analysis.learning_tips}</p>
          </div>
        </div>
      )}

      <AdMobBanner />
    </main>
  );
}
