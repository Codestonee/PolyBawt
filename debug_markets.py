"""Test the slug-based API endpoint for 15m markets."""
import asyncio
import aiohttp
import time
from datetime import datetime, timezone


async def test_slug_endpoint():
    """Test fetching 15m markets using slug endpoint."""
    
    async with aiohttp.ClientSession() as session:
        # Current time and recent 15m epochs
        now = int(time.time())
        
        # 15m markets likely use 15-minute aligned epochs
        # Round to nearest 15 minutes
        current_15m = (now // 900) * 900
        
        print(f"Current time: {datetime.fromtimestamp(now, tz=timezone.utc)}")
        print(f"Current 15m epoch: {current_15m} = {datetime.fromtimestamp(current_15m, tz=timezone.utc)}")
        
        # Try a few recent/upcoming epochs
        epochs_to_try = [
            current_15m + 900,  # Next 15m
            current_15m,        # Current 15m
            current_15m - 900,  # Previous 15m
            1769718600,         # From the screenshot
        ]
        
        for epoch in epochs_to_try:
            dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
            
            # Try different slug formats
            slug_formats = [
                f"btc-updown-15m-{epoch}",
                f"btc_updown_15m_{epoch}",
                f"btc-updown-15m",
                f"bitcoin-up-or-down-{epoch}",
            ]
            
            for slug in slug_formats:
                url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            print(f"\n=== FOUND: {slug} ===")
                            print(f"Title: {data.get('title')}")
                            print(f"Slug: {data.get('slug')}")
                            print(f"Markets: {len(data.get('markets', []))}")
                            for m in data.get('markets', [])[:3]:
                                print(f"  - Token IDs: {m.get('clobTokenIds')}")
                            return  # Found one!
                except Exception as e:
                    pass  # Continue trying
            
            # Also try the markets endpoint with the slug
            for slug in slug_formats[:2]:  # Just try the first two
                url = f"https://gamma-api.polymarket.com/markets"
                params = {"slug": slug}
                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data:
                                print(f"\n=== FOUND VIA MARKETS: {slug} ===")
                                print(f"Data: {data}")
                                return
                except:
                    pass
        
        print("\nNo 15m markets found via slug endpoints")
        
        # Let's try finding ANY event with similar patterns
        print("\n=== Trying to find pattern in events ===")
        url = "https://gamma-api.polymarket.com/events"
        async with session.get(url, params={"active": "true", "limit": 500}) as resp:
            events = await resp.json()
            
        # Look for crypto/btc related events
        for e in events:
            title = e.get("title", "").lower()
            slug = e.get("slug", "").lower()
            
            if any(x in title + slug for x in ["btc", "bitcoin", "eth", "sol", "xrp"]):
                if any(x in title + slug for x in ["up", "down", "price"]):
                    print(f"  - {e.get('slug')}: {e.get('title')[:60]}")


if __name__ == "__main__":
    asyncio.run(test_slug_endpoint())
