"use client";

import React from 'react';
import { useOrderStore, Order } from '@/stores/order-store';
import { XCircle, Trash2, Activity } from 'lucide-react';

export const ActiveOrders = () => {
    const { orders, cancelOrder, cancelAllOrders, addOrder, fetchOrders } = useOrderStore(); // Exposed for testing/mocking

    React.useEffect(() => {
        fetchOrders();
        const interval = setInterval(fetchOrders, 5000);
        return () => clearInterval(interval);
    }, [fetchOrders]);




    return (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg flex flex-col h-full min-h-[300px]">
            {/* Header */}
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
                <div className="flex items-center gap-2 text-zinc-100 font-semibold">
                    <Activity className="w-4 h-4 text-indigo-400" />
                    <span>Active Orders</span>
                    <span className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-full">{orders.length}</span>
                </div>

                {orders.length > 0 && (
                    <button
                        onClick={cancelAllOrders}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-500 text-xs font-medium rounded-md transition-colors border border-rose-500/20"
                    >
                        <Trash2 className="w-3 h-3" />
                        CANCEL ALL
                    </button>
                )}
            </div>

            {/* List */}
            <div className="flex-1 overflow-auto p-2 space-y-2">
                {orders.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-2">
                        <Activity className="w-8 h-8 opacity-20" />
                        <span className="text-sm">No active orders</span>
                    </div>
                ) : (
                    orders.map((order) => (
                        <div key={order.id} className="group flex items-center justify-between p-3 bg-zinc-950/50 hover:bg-zinc-800/50 border border-zinc-800/50 rounded-md transition-all">
                            <div className="flex flex-col">
                                <span className="font-bold text-zinc-200">{order.asset} <span className={order.side === 'YES' ? 'text-emerald-500' : 'text-rose-500'}>{order.side}</span></span>
                                <span className="text-xs text-zinc-500 font-mono">
                                    {(order.size).toLocaleString()} @ ${(order.price).toFixed(2)}
                                </span>
                            </div>

                            <div className="flex items-center gap-4">
                                <div className="text-right">
                                    <div className="text-xs text-zinc-400">Total</div>
                                    <div className="text-sm font-mono text-zinc-300">${(order.size * order.price).toFixed(2)}</div>
                                </div>
                                <button
                                    onClick={() => cancelOrder(order.id)}
                                    className="p-1.5 text-zinc-500 hover:text-rose-400 hover:bg-rose-400/10 rounded-md transition-colors opacity-0 group-hover:opacity-100"
                                    title="Cancel Order"
                                >
                                    <XCircle className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
