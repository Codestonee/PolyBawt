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

interface PortfolioState {
    balance: number
    equity: number
    positions: Position[]

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
                positions: data.positions
            });
        } catch (err) {
            console.error("Failed to fetch portfolio:", err);
        }
    }
}))
