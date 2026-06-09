"use client";
import React, { useState, useEffect, useRef } from "react";
import { useMarket } from "@/contexts/MarketContext";

interface Props {
  ticker: string;
  setTicker: (val: string) => void;
}

export default function StockSearch({ ticker, setTicker }: Props) {
  const { market } = useMarket();
  const [query, setQuery] = useState(ticker);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // market이 바뀔 때 query와 ticker 초기화
  useEffect(() => {
    setQuery("");
    setTicker("");
    setSuggestions([]);
    setIsOpen(false);
  }, [market, setTicker]);

  // Click outside to close
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (market !== "KR" || query.length < 1) {
      setSuggestions([]);
      return;
    }
    
    // 만약 이미 선택된 종목명과 같다면 검색 중지
    if (query === ticker) {
      setIsOpen(false);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(`http://localhost:8000/api/market/search?q=${encodeURIComponent(query)}`);
        if (res.ok) {
          const data = await res.json();
          setSuggestions(data.suggestions || []);
          setIsOpen(true);
        }
      } catch (err) {
        console.error("Search error", err);
      } finally {
        setLoading(false);
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timer);
  }, [query, market, ticker]);

  const handleSelect = (stockName: string) => {
    setQuery(stockName);
    setTicker(stockName);
    setIsOpen(false);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
    if (market === "US") {
      setTicker(e.target.value); // 미국장은 입력한 값이 곧 티커이므로 바로 반영
    } else {
      // 한국장은 드롭다운 클릭 전까지는 내부 상태 업데이트를 막음 (초기화)
      if (ticker) {
        setTicker("");
      }
    }
  };

  return (
    <div ref={wrapperRef} className="relative w-full">
      <input
        type="text"
        className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 bg-white"
        placeholder={market === "KR" ? "종목명 검색 (예: 삼성전자)" : "미국 주식 티커 직접 입력 (예: AAPL)"}
        value={query}
        onChange={handleChange}
        onFocus={() => {
          if (suggestions.length > 0 && query !== ticker) setIsOpen(true);
        }}
      />
      {loading && <div className="absolute right-4 top-3.5 text-slate-400 text-sm font-medium">검색중...</div>}
      
      {market === "KR" && isOpen && suggestions.length > 0 && (
        <ul className="absolute z-50 w-full mt-2 bg-white border border-slate-200 rounded-xl shadow-xl max-h-60 overflow-y-auto">
          {suggestions.map((s, idx) => (
            <li
              key={idx}
              className="px-4 py-3 hover:bg-indigo-50 cursor-pointer border-b border-slate-100 last:border-b-0 text-slate-700 font-medium transition-colors"
              onClick={() => handleSelect(s)}
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
