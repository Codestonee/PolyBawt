import { create } from 'zustand'

export interface Order {
    id: string
    asset: string
    side: 'YES' | 'NO'
    size: number
    price: number
    timestamp: Date
    status: 'OPEN' | 'FILLED' | 'CANCELLED'
}

interface OrderState {
    orders: Order[]

    // Actions
    addOrder: (order: Order) => void
    removeOrder: (id: string) => void
    cancelAllOrders: () => void
    cancelOrder: (id: string) => void
    fetchOrders: () => Promise<void>
}

export const useOrderStore = create<OrderState>((set) => ({
    orders: [],

    addOrder: (order) => set((state) => ({
        orders: [order, ...state.orders]
    })),

    removeOrder: (id) => set((state) => ({
        orders: state.orders.filter(o => o.id !== id)
    })),

    cancelAllOrders: () => set({ orders: [] }), // Mock implementation

    cancelOrder: (id) => set((state) => ({
        orders: state.orders.filter(o => o.id !== id)
    })),

    // API Integration
    fetchOrders: async () => {
        try {
            const res = await fetch('http://localhost:8000/api/orders');
            if (!res.ok) throw new Error('API Error');
            const data = await res.json();

            // Transform timestamp strings to Date objects if needed, 
            // or we expect frontend to handle ISO strings. 
            // Assuming simplified model for now.

            set({
                orders: data.map((o: any) => ({
                    ...o,
                    timestamp: new Date(o.timestamp || Date.now())
                }))
            });
        } catch (err) {
            console.error("Failed to fetch orders:", err);
        }
    }
}))
