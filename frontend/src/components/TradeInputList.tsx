"use client";
import React, { useState } from "react";
import { useMarket } from "@/contexts/MarketContext";

export interface TradePoint {
  id: string;
  type: "buy" | "sell";
  date: string;
  price: number;
  quantity: number;
}

interface Props {
  trades: TradePoint[];
  setTrades: React.Dispatch<React.SetStateAction<TradePoint[]>>;
}

export default function TradeInputList({ trades, setTrades }: Props) {
  const { currencySymbol, formatPrice } = useMarket();
  const [type, setType] = useState<"buy" | "sell">("buy");
  const [date, setDate] = useState("");
  const [price, setPrice] = useState("");
  const [quantity, setQuantity] = useState("");

  const addTrade = () => {
    if (!date || !price || !quantity) return;
    const newTrade: TradePoint = {
      id: Math.random().toString(36).substring(2, 9),
      type,
      date,
      price: Number(price),
      quantity: Number(quantity)
    };
    setTrades([...trades, newTrade]);
    setPrice("");
    setQuantity("");
  };

  const removeTrade = (id: string) => {
    setTrades(trades.filter(t => t.id !== id));
  };

  return (
    <div className="glass-card p-6 mb-6">
      <h3 className="text-lg font-bold text-slate-800 mb-4">매매 내역 입력</h3>
      
      <div className="flex gap-2 mb-4">
        <button 
          className={`flex-1 py-2.5 rounded-lg font-bold transition-colors ${type === "buy" ? "bg-red-500 text-white shadow-md shadow-red-500/30" : "bg-white text-slate-600 border border-slate-200"}`}
          onClick={() => setType("buy")}
        >🔴 매수</button>
        <button 
          className={`flex-1 py-2.5 rounded-lg font-bold transition-colors ${type === "sell" ? "bg-blue-500 text-white shadow-md shadow-blue-500/30" : "bg-white text-slate-600 border border-slate-200"}`}
          onClick={() => setType("sell")}
        >🔵 매도</button>
      </div>

      <div className="space-y-3">
        <input className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 bg-white" type="datetime-local" value={date} onChange={e => setDate(e.target.value)} />
        <input className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 bg-white" type="number" placeholder={`가격 (${currencySymbol})`} value={price} onChange={e => setPrice(e.target.value)} />
        <input className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 bg-white" type="number" placeholder="수량 (주)" value={quantity} onChange={e => setQuantity(e.target.value)} />
      </div>

      <button onClick={addTrade} className="btn-primary mt-4">
        + 추가하기
      </button>

      {trades.length > 0 && (
        <div className="mt-6 space-y-3">
          {trades.map(trade => (
            <div key={trade.id} className={`flex justify-between items-center p-4 bg-white rounded-xl shadow-sm border-l-4 ${trade.type === "buy" ? "border-red-500" : "border-blue-500"}`}>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-slate-500 font-medium">{trade.type === "buy" ? "매수" : "매도"} • {new Date(trade.date).toLocaleString()}</span>
                <span className="text-base font-bold text-slate-800">{currencySymbol}{formatPrice(trade.price)}</span>
                <span className="text-sm text-slate-600">{trade.quantity}주</span>
              </div>
              <button onClick={() => removeTrade(trade.id)} className="p-2 text-slate-400 hover:text-red-500 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
