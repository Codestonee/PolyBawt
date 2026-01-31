import { create } from 'zustand'

export interface Position {
    asset: string
    side: 'YES' | 'NO'
    size: number
    avgPrice: number
    currentPrice: number
    pnl: number
    pnlPercent: number
}

export interface WinRate {
    total_trades: number
    total_wins: number
    total_losses: number
    win_rate: number
    current_streak: number
    max_consecutive_wins: number
    max_consecutive_losses: number
    biggest_win: number
    biggest_loss: number
    average_win: number
    average_loss: number
    profit_factor: number | string
}

interface PortfolioState {
    balance: number
    equity: number
    positions: Position[]
    winRate: WinRate | null

    // Actions
    updateBalance: (balance: number) => void
    updateEquity: (equity: number) => void
    updatePositions: (positions: Position[]) => void
    fetchPortfolio: () => Promise<void>

    // Helpers
    getTotalPnl: () => number
}

export const usePortfolioStore = create<PortfolioState>((set, get) => ({
    balance: 0.00, // Default to 0 until data loads
    equity: 0.00,
    positions: [],
    winRate: null,

    updateBalance: (balance) => set({ balance }),
    updateEquity: (equity) => set({ equity }),
    updatePositions: (positions) => set({ positions }),

    getTotalPnl: () => get().equity - 0.00, // Base also 0

    // API Integration
    fetchPortfolio: async () => {
        try {
            const res = await fetch('http://localhost:8000/api/portfolio');
            if (!res.ok) throw new Error('API Error');
            const data = await res.json();

            set({
                balance: data.balance,
                equity: data.equity,
                positions: data.positions,
                winRate: data.win_rate
            });
        } catch (err) {
            console.error("Failed to fetch portfolio:", err);
        }
    }
}))
