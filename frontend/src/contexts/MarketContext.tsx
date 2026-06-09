"use client";
import React, { createContext, useContext, useState, ReactNode } from "react";

type MarketType = "KR" | "US";

interface MarketContextType {
  market: MarketType;
  setMarket: (m: MarketType) => void;
  currencySymbol: string;
  formatPrice: (price: number) => string;
}

const MarketContext = createContext<MarketContextType | undefined>(undefined);

export function MarketProvider({ children }: { children: ReactNode }) {
  const [market, setMarket] = useState<MarketType>("KR");

  const currencySymbol = market === "KR" ? "₩" : "$";

  const formatPrice = (price: number) => {
    if (market === "KR") {
      return price.toLocaleString("ko-KR");
    } else {
      return price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
  };

  return (
    <MarketContext.Provider value={{ market, setMarket, currencySymbol, formatPrice }}>
      {children}
    </MarketContext.Provider>
  );
}

export function useMarket() {
  const context = useContext(MarketContext);
  if (context === undefined) {
    throw new Error("useMarket must be used within a MarketProvider");
  }
  return context;
}
