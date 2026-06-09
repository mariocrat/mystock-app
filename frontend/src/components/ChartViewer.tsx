"use client";
import React, { useEffect, useRef } from "react";
import { createChart, IChartApi, ISeriesApi } from "lightweight-charts";
import { TradePoint } from "./TradeInputList";

interface ChartViewerProps {
  data: any[];
  trades?: TradePoint[];
  market?: "KR" | "US";
}

export default function ChartViewer({ data, trades = [], market = "KR" }: ChartViewerProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    // Convert time to unix timestamp for TradingView
    const formattedData = data.map(d => ({
      time: new Date(d.time).getTime() / 1000,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      value: d.value || 0
    })).sort((a, b) => a.time - b.time);

    const timezone = market === "US" ? "America/New_York" : "Asia/Seoul";

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: "transparent" },
        textColor: "#333",
      },
      grid: {
        vertLines: { color: "#f1f5f9" },
        horzLines: { color: "#f1f5f9" },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: number) => {
          return new Intl.DateTimeFormat('ko-KR', {
            timeZone: timezone,
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
          }).format(new Date(time * 1000));
        },
      },
      localization: {
        timeFormatter: (time: number) => {
          return new Intl.DateTimeFormat('ko-KR', {
            timeZone: timezone,
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
          }).format(new Date(time * 1000));
        }
      }
    });
    
    chartRef.current = chart;

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#ef4444', 
      downColor: '#3b82f6', 
      borderVisible: false,
      wickUpColor: '#ef4444', 
      wickDownColor: '#3b82f6'
    });
    seriesRef.current = candlestickSeries;
    candlestickSeries.setData(formattedData as any);

    // Add Volume Series
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });
    
    const volumeData = formattedData.map(d => ({
      time: d.time,
      value: d.value,
      color: d.close >= d.open ? 'rgba(239, 68, 68, 0.4)' : 'rgba(59, 130, 246, 0.4)'
    }));
    volumeSeries.setData(volumeData as any);

    if (trades.length > 0) {
      const markers = trades.map(trade => {
        const matchedCandle = data.find(d => d.time.startsWith(trade.date));
        const unixTime = matchedCandle 
          ? new Date(matchedCandle.time).getTime() / 1000 
          : new Date(trade.date).getTime() / 1000;

        return {
          time: unixTime,
          position: trade.type === "buy" ? "belowBar" : "aboveBar",
          color: trade.type === "buy" ? "#EF4444" : "#3B82F6",
          shape: trade.type === "buy" ? "arrowUp" : "arrowDown",
          text: trade.type === "buy" ? "B" : "S",
        };
      }).sort((a, b) => a.time - b.time);
      
      candlestickSeries.setMarkers(markers as any);
    }

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data]);

  if (data.length === 0) return null;

  return (
    <div className="glass-card p-4 mb-6">
      <h3 className="text-lg font-bold text-slate-800 mb-4 px-2">시계열 차트</h3>
      <div ref={chartContainerRef} className="w-full rounded-xl overflow-hidden" />
    </div>
  );
}
