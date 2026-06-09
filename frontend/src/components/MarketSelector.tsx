"use client";
import React from "react";
import { useMarket } from "@/contexts/MarketContext";

export default function MarketSelector() {
  const { market, setMarket } = useMarket();

  return (
    <div className="flex flex-col items-center mb-6">
      <div className="flex gap-3 w-full bg-slate-100 p-1.5 rounded-2xl shadow-inner">
        <button 
          className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all duration-300 ${market === "KR" ? "bg-white text-indigo-600 shadow-md ring-1 ring-indigo-100" : "text-slate-500 hover:bg-slate-200/50"}`} 
          onClick={() => setMarket("KR")}
        >
          🇰🇷 한국 (KST)
        </button>
        <button 
          className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all duration-300 ${market === "US" ? "bg-white text-indigo-600 shadow-md ring-1 ring-indigo-100" : "text-slate-500 hover:bg-slate-200/50"}`} 
          onClick={() => setMarket("US")}
        >
          🇺🇸 미국 (ET)
        </button>
      </div>
      <p className="text-sm mt-3 text-slate-500 font-medium">
        {market === "KR" ? "ℹ️ 국장: 수익률 계산 시 수수료 및 세금이 일부 반영된 기준입니다." : "ℹ️ 미장: 단순 평단가 기준입니다. (수수료/세금 미반영)"}
      </p>
    </div>
  );
}
