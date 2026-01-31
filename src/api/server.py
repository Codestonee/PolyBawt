from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.state import get_state
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


# --- Data Models ---

class PositionModel(BaseModel):
    asset: str
    side: str
    size: float
    entry_price: float
    current_value: float
    pnl: float
    pnl_percent: float


class WinRateModel(BaseModel):
    """Win rate and trading statistics."""
    total_trades: int
    total_wins: int
    total_losses: int
    win_rate: float  # Percentage (0-100)
    current_streak: int  # Positive = wins, negative = losses
    max_consecutive_wins: int
    max_consecutive_losses: int
    biggest_win: float
    biggest_loss: float
    average_win: float
    average_loss: float
    profit_factor: float | str  # Can be "inf" if no losses


class PortfolioModel(BaseModel):
    balance: float
    equity: float
    total_exposure: float
    daily_pnl: float
    daily_return_pct: float
    positions: List[PositionModel]
    win_rate: WinRateModel | None = None

    # Safety/UX: these help the dashboard avoid showing fake precision.
    unrealized_pnl_available: bool = False
    note: str | None = None


class OrderModel(BaseModel):
    id: str
    asset: str
    side: str
    size: float
    price: float
    status: str
    timestamp: str


# --- API Implementation ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events."""
    logger.info("API Server starting up")
    yield
    logger.info("API Server shutting down")


app = FastAPI(title="Polymarket Bot API", lifespan=lifespan)

# CORS Config - Allow frontend local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API root - lists available endpoints."""
    state = get_state()
    return {
        "name": "Polymarket Bot API",
        "version": "1.0.0",
        "status": "online",
        "bot_running": state.is_running,
        "is_live": state.is_live,
        "endpoints": {
            "health": "/health",
            "portfolio": "/api/portfolio",
            "stats": "/api/stats",
            "orders": "/api/orders"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    state = get_state()
    return {
        "status": "online",
        "bot_running": state.is_running,
        "is_live": state.is_live,
        "active_strategies": state.active_strategies,
        "last_update": state.last_update
    }


@app.get("/api/portfolio", response_model=PortfolioModel)
async def get_portfolio():
    """Get current portfolio state."""
    state = get_state()
    pf = state.portfolio
    
    if not pf:
        # Return empty state if not initialized
        return PortfolioModel(
            balance=0,
            equity=0,
            total_exposure=0,
            daily_pnl=0,
            daily_return_pct=0,
            positions=[],
            win_rate=WinRateModel(
                total_trades=0,
                total_wins=0,
                total_losses=0,
                win_rate=0.0,
                current_streak=0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                biggest_win=0.0,
                biggest_loss=0.0,
                average_win=0.0,
                average_loss=0.0,
                profit_factor=0.0,
            )
        )
    
    # Convert Portfolio object to API model
    summary = pf.summary()
    
    # Convert positions
    # Note: Accessing private _positions for read-only API view
    # In a stricter imp, we'd add public getters to Portfolio
    positions_data = []
    
    # We need current prices to calc pnl properly, but for now we use what's tracked
    # Ideally, Portfolio should have the latest mark prices.
    # For now, we return the tracked structure.
    
    # We currently do NOT have reliable mark prices available at the API layer.
    # So we report position value at entry (not "current") and set pnl to 0.
    for p in pf.get_open_positions():
        current_val = p.current_value
        pnl = 0.0
        pnl_pct = 0.0

        positions_data.append(PositionModel(
            asset=p.asset,
            side=p.side.value if hasattr(p.side, 'value') else str(p.side),
            size=p.size_usd,
            entry_price=p.entry_price,
            current_value=current_val,
            pnl=pnl,
            pnl_percent=pnl_pct,
        ))

    # Build win rate stats
    win_rate_data = WinRateModel(
        total_trades=summary.get("total_trades", 0),
        total_wins=summary.get("total_wins", 0),
        total_losses=summary.get("total_losses", 0),
        win_rate=summary.get("win_rate", 0.0),
        current_streak=summary.get("current_streak", 0),
        max_consecutive_wins=summary.get("max_consecutive_wins", 0),
        max_consecutive_losses=summary.get("max_consecutive_losses", 0),
        biggest_win=summary.get("biggest_win", 0.0),
        biggest_loss=summary.get("biggest_loss", 0.0),
        average_win=summary.get("average_win", 0.0),
        average_loss=summary.get("average_loss", 0.0),
        profit_factor=summary.get("profit_factor", 0.0),
    )

    return PortfolioModel(
        balance=pf.current_capital,
        # Without reliable mark prices, equity == balance.
        equity=pf.current_capital,
        total_exposure=pf.total_exposure,
        daily_pnl=summary.get("daily_pnl", 0),
        daily_return_pct=summary.get("daily_return_pct", 0),
        positions=positions_data,
        win_rate=win_rate_data,
        unrealized_pnl_available=False,
        note="Unrealized PnL/equity marks not available (no live mark prices wired to API).",
    )


@app.get("/api/stats", response_model=WinRateModel)
async def get_stats():
    """Get trading statistics including win rate."""
    state = get_state()
    pf = state.portfolio

    if not pf:
        return WinRateModel(
            total_trades=0,
            total_wins=0,
            total_losses=0,
            win_rate=0.0,
            current_streak=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            biggest_win=0.0,
            biggest_loss=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
        )

    summary = pf.summary()
    return WinRateModel(
        total_trades=summary.get("total_trades", 0),
        total_wins=summary.get("total_wins", 0),
        total_losses=summary.get("total_losses", 0),
        win_rate=summary.get("win_rate", 0.0),
        current_streak=summary.get("current_streak", 0),
        max_consecutive_wins=summary.get("max_consecutive_wins", 0),
        max_consecutive_losses=summary.get("max_consecutive_losses", 0),
        biggest_win=summary.get("biggest_win", 0.0),
        biggest_loss=summary.get("biggest_loss", 0.0),
        average_win=summary.get("average_win", 0.0),
        average_loss=summary.get("average_loss", 0.0),
        profit_factor=summary.get("profit_factor", 0.0),
    )


@app.get("/api/orders", response_model=List[OrderModel])
async def get_orders():
    """Get active orders."""
    state = get_state()
    # Convert internal Order objects to API models
    api_orders = []
    for o in state.active_orders:
        # Safely extract attributes, internal structure might vary
        api_orders.append(OrderModel(
            id=getattr(o, "client_order_id", "unknown"),
            asset=getattr(o, "asset", "unknown"),
            side=str(getattr(o, "side", "unknown")),
            size=getattr(o, "size", 0.0),
            price=getattr(o, "price", 0.0),
            status=getattr(o, "state", "unknown").name if hasattr(getattr(o, "state", None), "name") else str(getattr(o, "state", "unknown")),
            timestamp=str(getattr(o, "timestamp", ""))
        ))
    return api_orders


# Entry point for running standalone
if __name__ == "__main__":
    import uvicorn
    print("🌐 Starting Polymarket Bot API Server...")
    print("📡 API: http://localhost:8000")
    print("📊 Health: http://localhost:8000/health")
    print("💼 Portfolio: http://localhost:8000/api/portfolio")
    print("📈 Stats: http://localhost:8000/api/stats")
    print("📋 Orders: http://localhost:8000/api/orders")
    print("\n⚠️  Note: Run with main bot for live data updates")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
