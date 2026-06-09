import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { MarketProvider } from "@/contexts/MarketContext";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI 주식 매매 복기",
  description: "AI Stock Trading Review App",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body className={inter.className}>
        <MarketProvider>
          {children}
        </MarketProvider>
      </body>
    </html>
  );
}
