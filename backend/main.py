from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import market, analysis
import uvicorn

app = FastAPI(title="Stock Training API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Stock Training API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
