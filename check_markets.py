"""Quick script to check what markets are available."""
import asyncio
from src.ingestion.market_discovery import MarketDiscovery


async def check_markets():
    discovery = MarketDiscovery()
    
    # Get all crypto markets (tag_id=21)
    print("Fetching crypto markets from Gamma API...")
    raw_markets = await discovery.get_markets(active=True, tag_id=21, limit=20)
    
    print(f"\nTotal crypto markets found: {len(raw_markets)}")
    print("\nSample questions:")
    for m in raw_markets[:10]:
        q = m.get("question", "")[:80]
        end = m.get("endDate", "")[:19]
        print(f"  - [{end}] {q}")
    
    # Check for 15m markets specifically
    print("\n\nLooking for 15m markets...")
    markets_15m = await discovery.get_crypto_15m_markets()
    print(f"15m markets found: {len(markets_15m)}")
    
    for m in markets_15m[:5]:
        print(f"  - {m.asset}: {m.question[:60]}... (expires in {m.minutes_to_expiry:.1f}m)")
    
    await discovery.close()


if __name__ == "__main__":
    asyncio.run(check_markets())
