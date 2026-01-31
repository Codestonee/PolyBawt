# Polymarket Bot Real-Time Dashboard

A professional, real-time trading operations dashboard for the Polymarket trading bot.

## Features

### Real-Time Data Updates
- **Auto-polling**: Fetches data every 2 seconds when tab is active
- **Connection status**: Visual indicators for API connection quality
- **Live pulse indicators**: Animated indicators show active data flow
- **Stale data warnings**: Alerts when connection becomes stale

### Portfolio Metrics
- Balance and equity display
- Daily P&L with trend indicators
- Win rate and streak tracking
- Exposure and margin usage
- Profit factor analysis
- Trading mode indicator (Live/Paper)

### Active Positions Table
- Sortable columns (Asset, Size, Entry Price, Current Price, P&L)
- Side filtering (YES/NO positions)
- Real-time P&L calculation
- Color-coded profit/loss indicators
- Responsive design with truncation

### Live Orders Panel
- Real-time order feed
- Status indicators (Pending, Filled, Partial, Cancelled, Rejected)
- Progress bars for partially filled orders
- Timestamp relative formatting
- Most recent 10 orders display

### Performance Chart
- Equity curve visualization
- Time range selector (1D, 1W, 1M, 3M, ALL)
- Interactive tooltips with P&L details
- Gradient area chart with reference line
- Auto-refreshing chart data

### Market Microstructure Monitor
- Real-time VPIN (Volume-synchronized Probability of Informed Trading)
- OBI (Order Book Imbalance) signals
- Market toxicity assessment
- 24h volume tracking
- Price spread monitoring
- Color-coded risk levels

### System Health Panel
- API Server connection latency
- WebSocket status
- Database health
- CLOB (Central Limit Order Book) connection
- Connection quality indicator
- Circuit breaker status with progress bars

### Strategy Status Panel
- Individual strategy toggles
- Weight allocation visualization
- Trade count per strategy
- Win rate per strategy
- Last trade timestamps
- Combined Sharpe ratio display

## Architecture

### State Management
- **Zustand store** (`realtimeStore.ts`):
  - Polling-based data fetching
  - Automatic connection recovery
  - Tab visibility awareness
  - Type-safe state management

### API Integration
- FastAPI backend integration
- Endpoints:
  - `/health` - Bot status and active strategies
  - `/api/portfolio` - Portfolio balance, equity, positions
  - `/api/orders` - Active orders list
  - `/api/stats` - Trading statistics

### Component Structure
```
components/dashboard/
├── MetricCard.tsx          # Reusable metric display card
├── PortfolioMetrics.tsx    # 8-metric grid layout
├── PositionsTable.tsx      # Sortable positions table
├── LiveOrders.tsx          # Real-time order feed
├── PerformanceChart.tsx    # Equity curve chart
├── MarketMicrostructure.tsx # Market data analysis
├── SystemHealth.tsx        # System status panel
└── StrategyStatus.tsx      # Strategy management
```

## Design System

### Colors
- **Background**: `#121212` (Dark charcoal)
- **Surface 1**: `#1E1E1E` (Main panels)
- **Surface 2**: `#252525` (Hover states)
- **Border**: `#333333` (Subtle borders)
- **Text Primary**: `#E0E0E0` (90% white)
- **Text Secondary**: `#A0A0A0` (60% white)
- **Success**: `#10B981` (Green)
- **Warning**: `#F59E0B` (Amber)
- **Danger**: `#EF4444` (Red)
- **Info**: `#3B82F6` (Blue)

### Typography
- Font family: System sans-serif with tabular numbers
- Tabular numerics for financial data
- Clear visual hierarchy with size and weight

### Animations
- Pulse animations for live indicators
- Smooth transitions on hover
- Chart animations for data updates
- Progress bar transitions

## Development

### Running the Dashboard

1. Start the API server:
```bash
cd polymarket-bot
python -m src.api.server
```

2. Start the Next.js dev server:
```bash
cd web-client
npm run dev
```

3. Open http://localhost:3000

### Building for Production
```bash
npm run build
```

### Key Dependencies
- Next.js 16.1.6
- React 19.2.3
- TypeScript
- Tailwind CSS 4
- Zustand 5.0.10
- Recharts 3.7.0
- Lucide React icons
- date-fns

## Future Enhancements

- WebSocket support for true real-time updates
- Historical trade analysis
- Advanced charting with technical indicators
- Strategy backtesting visualization
- Risk metrics heatmap
- Export functionality for reports
- Mobile-responsive optimizations
