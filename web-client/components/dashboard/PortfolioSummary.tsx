"use client";

import React from 'react';
import { usePortfolioStore } from '@/stores/portfolio-store';
import { ArrowUpRight, ArrowDownRight, Wallet, PieChart } from 'lucide-react';

export const PortfolioSummary = () => {
    const { balance, equity, winRate, getTotalPnl, fetchPortfolio } = usePortfolioStore();
    const pnl = getTotalPnl();
    const pnlPercent = (pnl / (equity - pnl || 1)) * 100;
    const isProfitable = pnl >= 0;

    React.useEffect(() => {
        fetchPortfolio();
        const interval = setInterval(fetchPortfolio, 5000);
        return () => clearInterval(interval);
    }, [fetchPortfolio]);

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Balance Card */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 flex flex-col justify-between h-[120px]">
                <div className="flex items-center gap-2 text-zinc-400 text-sm font-medium uppercase tracking-wider">
                    <Wallet className="w-4 h-4" />
                    <span>Available Balance</span>
                </div>
                <div className="text-3xl font-mono font-bold text-white tracking-tight">
                    ${balance.toLocaleString('en-US', { minimumFractionDigits: 2 })} <span className="text-base text-zinc-500 font-normal">USDC</span>
                </div>
            </div>

            {/* Equity Card */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 flex flex-col justify-between h-[120px]">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-zinc-400 text-sm font-medium uppercase tracking-wider">
                        <PieChart className="w-4 h-4" />
                        <span>Total Equity</span>
                    </div>
                    <div className={`flex items-center gap-1 text-sm font-mono ${isProfitable ? 'text-emerald-500' : 'text-rose-500'}`}>
                        {isProfitable ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                        <span>{Math.abs(pnlPercent).toFixed(2)}%</span>
                    </div>
                </div>
                <div className="flex items-end gap-3">
                    <div className="text-3xl font-mono font-bold text-white tracking-tight">
                        ${equity.toLocaleString('en-US', { minimumFractionDigits: 2 })} <span className="text-base text-zinc-500 font-normal">USDC</span>
                    </div>
                </div>
            </div>

            {/* Win Rate Card */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 flex flex-col justify-between h-[120px]">
                <div className="flex items-center gap-2 text-zinc-400 text-sm font-medium uppercase tracking-wider">
                    <span className="text-lg">🎯</span>
                    <span>Win Rate</span>
                </div>
                <div className="text-3xl font-mono font-bold text-white tracking-tight">
                    {winRate ? winRate.win_rate.toFixed(1) : "0.0"}%
                </div>
                <div className="text-xs text-zinc-500 font-mono">
                    {winRate ? `${winRate.total_wins}W - ${winRate.total_losses}L` : "No trades"}
                </div>
            </div>

            {/* Profit Factor Card */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 flex flex-col justify-between h-[120px]">
                <div className="flex items-center gap-2 text-zinc-400 text-sm font-medium uppercase tracking-wider">
                    <span className="text-lg">⚖️</span>
                    <span>Profit Factor</span>
                </div>
                <div className="text-3xl font-mono font-bold text-white tracking-tight">
                    {winRate ? winRate.profit_factor : "0.0"}
                </div>
                <div className="text-xs text-zinc-500 font-mono">
                    Expectancy: {winRate ? `$${(winRate.average_win - winRate.average_loss).toFixed(2)}` : "$0.00"}
                </div>
            </div>
        </div>
    );
};
