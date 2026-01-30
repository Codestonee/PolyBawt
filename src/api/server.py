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


class PortfolioModel(BaseModel):
    balance: float
    equity: float
    total_exposure: float
    daily_pnl: float
    daily_return_pct: float
    positions: List[PositionModel]


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


@app.get("/health")
async def health_check():
    state = get_state()
    return {
        "status": "online",
        "bot_running": state.is_running,
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
            positions=[]
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
    
    for p in pf.get_open_positions():
        # simple estimation if we don't have live mark price in the object
        # The portfolio object relies on settle_position or unrealized_pnl(prices) 
        # API layer might not have fresh prices immediately available unless shared
        current_val = p.size_usd # Placeholder if we don't have live mark
        pnl = 0.0
        pnl_pct = 0.0
        
        positions_data.append(PositionModel(
            asset=p.asset,
            side=p.side.value if hasattr(p.side, 'value') else str(p.side),
            size=p.size_usd,
            entry_price=p.entry_price,
            current_value=current_val,
            pnl=pnl,
            pnl_percent=pnl_pct
        ))

    return PortfolioModel(
        balance=pf.current_capital,
        equity=pf.current_capital + pf.unrealized_pnl({}), # PnL needs prices
        total_exposure=pf.total_exposure,
        daily_pnl=summary.get("daily_pnl", 0),
        daily_return_pct=summary.get("daily_return_pct", 0),
        positions=positions_data
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
