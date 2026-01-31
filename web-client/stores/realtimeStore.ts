/**
 * Real-time data store with WebSocket integration
 * Manages live connection to bot API for instant updates
 */

import { create } from 'zustand';

// Types
export interface Position {
  id: string;
  asset: string;
  side: 'long_yes' | 'long_no';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  openedAt: string;
  marketQuestion: string;
  category?: string;
}

export interface Order {
  id: string;
  asset: string;
  side: 'buy' | 'sell';
  size: number;
  price: number;
  status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'rejected';
  filledSize: number;
  timestamp: string;
  marketQuestion?: string;
}

export interface MarketData {
  asset: string;
  yesPrice: number;
  noPrice: number;
  spread: number;
  volume24h: number;
  liquidity: number;
  timeToExpiry: number;
  obiSignal: number; // -1 to +1
  vpin: number;
  toxicity: 'low' | 'moderate' | 'high' | 'extreme';
}

export interface PerformanceMetrics {
  totalTrades: number;
  winRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  dailyPnl: number;
  dailyReturn: number;
  brierScore: number;
  currentStreak: number;
}

export interface CircuitBreakerStatus {
  type: string;
  state: 'closed' | 'soft' | 'hard';
  currentValue: number;
  threshold: number;
  message: string;
  trippedAt?: string;
}

export interface BotStatus {
  isRunning: boolean;
  isLive: boolean;
  mode: 'paper' | 'live';
  uptime: number;
  lastUpdate: string;
  activeStrategies: Record<string, boolean>;
}

interface RealtimeState {
  // Connection
  connected: boolean;
  lastPing: number;
  reconnectAttempts: number;
  
  // Data
  portfolio: {
    balance: number;
    equity: number;
    totalExposure: number;
    availableCash: number;
    marginUsed: number;
  };
  positions: Position[];
  orders: Order[];
  markets: MarketData[];
  performance: PerformanceMetrics;
  circuitBreakers: CircuitBreakerStatus[];
  botStatus: BotStatus;
  recentTrades: Array<{
    timestamp: string;
    asset: string;
    side: string;
    size: number;
    price: number;
    pnl?: number;
  }>;
  
  // Event Markets
  eventMarkets: Array<{
    id: string;
    question: string;
    category: string;
    yesPrice: number;
    probability: number;
    edge: number;
    confidence: number;
  }>;
  
  // Actions
  connect: () => void;
  disconnect: () => void;
  fetchInitialData: () => Promise<void>;
  refreshData: () => Promise<void>;
}

// API base URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Polling interval (ms) - fallback if WebSocket not available
const POLL_INTERVAL = 2000;

export const useRealtimeStore = create<RealtimeState>((set, get) => ({
  // Initial state
  connected: false,
  lastPing: Date.now(),
  reconnectAttempts: 0,
  
  portfolio: {
    balance: 0,
    equity: 0,
    totalExposure: 0,
    availableCash: 0,
    marginUsed: 0,
  },
  positions: [],
  orders: [],
  markets: [],
  performance: {
    totalTrades: 0,
    winRate: 0,
    profitFactor: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    dailyPnl: 0,
    dailyReturn: 0,
    brierScore: 0,
    currentStreak: 0,
  },
  circuitBreakers: [],
  botStatus: {
    isRunning: false,
    isLive: false,
    mode: 'paper',
    uptime: 0,
    lastUpdate: new Date().toISOString(),
    activeStrategies: {},
  },
  recentTrades: [],
  eventMarkets: [],
  
  // Actions
  connect: () => {
    // Initial data fetch
    get().fetchInitialData();
    
    // Set up polling (WebSocket can be added later)
    const interval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        get().refreshData();
      }
    }, POLL_INTERVAL);
    
    // Store interval for cleanup
    (window as any).__dashboardInterval = interval;
    
    set({ connected: true, lastPing: Date.now() });
  },
  
  disconnect: () => {
    if ((window as any).__dashboardInterval) {
      clearInterval((window as any).__dashboardInterval);
    }
    set({ connected: false });
  },
  
  fetchInitialData: async () => {
    try {
      // Fetch all initial data in parallel
      const [healthRes, portfolioRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/health`),
        fetch(`${API_BASE}/api/portfolio`),
        fetch(`${API_BASE}/api/stats`),
      ]);
      
      const health = await healthRes.json();
      const portfolio = await portfolioRes.json();
      const stats = await statsRes.json();
      
      set({
        botStatus: {
          isRunning: health.bot_running,
          isLive: health.is_live,
          mode: health.is_live ? 'live' : 'paper',
          uptime: 0,
          lastUpdate: health.last_update || new Date().toISOString(),
          activeStrategies: health.active_strategies || {},
        },
        portfolio: {
          balance: portfolio.balance,
          equity: portfolio.equity,
          totalExposure: portfolio.total_exposure,
          availableCash: portfolio.balance - portfolio.total_exposure,
          marginUsed: portfolio.total_exposure / portfolio.balance * 100,
        },
        positions: portfolio.positions.map((p: any) => ({
          id: `${p.asset}-${p.side}`,
          asset: p.asset,
          side: p.side,
          size: p.size,
          entryPrice: p.entry_price,
          currentPrice: p.current_value / p.size || p.entry_price,
          pnl: p.pnl,
          pnlPercent: p.pnl_percent,
          openedAt: new Date().toISOString(),
          marketQuestion: p.asset,
        })),
        performance: {
          totalTrades: stats.total_trades,
          winRate: stats.win_rate,
          profitFactor: typeof stats.profit_factor === 'number' ? stats.profit_factor : 0,
          sharpeRatio: 0, // TODO
          maxDrawdown: 0, // TODO
          dailyPnl: portfolio.daily_pnl,
          dailyReturn: portfolio.daily_return_pct,
          brierScore: 0, // TODO
          currentStreak: stats.current_streak,
        },
      });
    } catch (error) {
      console.error('Failed to fetch initial data:', error);
    }
  },
  
  refreshData: async () => {
    try {
      // Fetch only essential updates
      const [portfolioRes, ordersRes] = await Promise.all([
        fetch(`${API_BASE}/api/portfolio`),
        fetch(`${API_BASE}/api/orders`),
      ]);
      
      const portfolio = await portfolioRes.json();
      const orders = await ordersRes.json();
      
      set(state => ({
        ...state,
        portfolio: {
          balance: portfolio.balance,
          equity: portfolio.equity,
          totalExposure: portfolio.total_exposure,
          availableCash: portfolio.balance - portfolio.total_exposure,
          marginUsed: portfolio.total_exposure / portfolio.balance * 100,
        },
        positions: portfolio.positions.map((p: any) => ({
          id: `${p.asset}-${p.side}`,
          asset: p.asset,
          side: p.side,
          size: p.size,
          entryPrice: p.entry_price,
          currentPrice: p.current_value / p.size || p.entry_price,
          pnl: p.pnl,
          pnlPercent: p.pnl_percent,
          openedAt: new Date().toISOString(),
          marketQuestion: p.asset,
        })),
        orders: orders.map((o: any) => ({
          id: o.id,
          asset: o.asset,
          side: o.side.toLowerCase(),
          size: o.size,
          price: o.price,
          status: o.status.toLowerCase(),
          filledSize: 0,
          timestamp: o.timestamp,
        })),
        lastPing: Date.now(),
      }));
    } catch (error) {
      console.error('Failed to refresh data:', error);
      set(state => ({
        ...state,
        reconnectAttempts: state.reconnectAttempts + 1,
      }));
    }
  },
}));
