import { create } from 'zustand'

export type RiskLevel = 'normal' | 'de-risk' | 'flatline'

interface RiskState {
    // Risk Controls
    riskLevel: 'normal' | 'de-risk' | 'flatline'
    setRiskLevel: (level: 'normal' | 'de-risk' | 'flatline') => void

    isShadowMode: boolean
    toggleShadowMode: () => void

    // Replay State
    currentTime: Date | null
    setCurrentTime: (time: Date | null) => void

    // PreFlight State
    isPreFlightOpen: boolean
    setPreFlightOpen: (open: boolean) => void

    // UI Configuration
    density: 'comfortable' | 'compact'
    setDensity: (density: 'comfortable' | 'compact') => void
}

export const useRiskStore = create<RiskState>((set) => ({
    riskLevel: 'normal',
    setRiskLevel: (level) => set({ riskLevel: level }),

    isShadowMode: true, // Default to shadow for safety
    toggleShadowMode: () => set((state) => ({ isShadowMode: !state.isShadowMode })),

    currentTime: null,
    setCurrentTime: (time) => set({ currentTime: time }),

    isPreFlightOpen: false,
    setPreFlightOpen: (open) => set({ isPreFlightOpen: open }),

    density: 'comfortable',
    setDensity: (density) => set({ density }),
}))
