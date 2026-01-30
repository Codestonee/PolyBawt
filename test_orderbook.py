"""Debug order book for both YES and NO tokens."""
import asyncio
from py_clob_client.client import ClobClient
import aiohttp
import time
import json


async def debug_both_tokens():
    """Check order books for both YES and NO tokens."""
    
    print("Creating read-only CLOB client...")
    client = ClobClient(host="https://clob.polymarket.com")
    
    now = int(time.time())
    current_15m = (now // 900) * 900
    
    async with aiohttp.ClientSession() as session:
        slug = f"btc-updown-15m-{current_15m}"
        url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
        
        async with session.get(url) as resp:
            if resp.status != 200:
                slug = f"btc-updown-15m-{current_15m + 900}"
                url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
                async with session.get(url) as resp2:
                    event = await resp2.json()
            else:
                event = await resp.json()
        
        markets = event.get("markets", [])
        if not markets:
            print("No markets")
            return
        
        market = markets[0]
        token_ids_raw = market.get("clobTokenIds", [])
        outcomes_raw = market.get("outcomes", [])
        
        # Parse
        if isinstance(token_ids_raw, str):
            token_ids = json.loads(token_ids_raw)
        else:
            token_ids = token_ids_raw
            
        if isinstance(outcomes_raw, str):
            outcomes = json.loads(outcomes_raw)
        else:
            outcomes = outcomes_raw
        
        print(f"Market: {event.get('title')}")
        print(f"Outcomes: {outcomes}")
        print(f"Token count: {len(token_ids)}")
        
        for i, token_id in enumerate(token_ids):
            outcome = outcomes[i] if i < len(outcomes) else f"Token {i}"
            print(f"\n{'='*50}")
            print(f"Outcome: {outcome}")
            print(f"Token ID: {token_id[:40]}...")
            
            try:
                book = client.get_order_book(token_id)
                
                print(f"Bids: {len(book.bids)}, Asks: {len(book.asks)}")
                
                if book.bids:
                    # Best bid is highest price
                    best_bid = max(book.bids, key=lambda x: float(x.price))
                    print(f"Best BID: ${float(best_bid.price):.4f} (size: {float(best_bid.size):.2f})")
                
                if book.asks:
                    # Best ask is lowest price
                    best_ask = min(book.asks, key=lambda x: float(x.price))
                    print(f"Best ASK: ${float(best_ask.price):.4f} (size: {float(best_ask.size):.2f})")
                    
                if book.bids and book.asks:
                    best_bid = max(book.bids, key=lambda x: float(x.price))
                    best_ask = min(book.asks, key=lambda x: float(x.price))
                    spread = float(best_ask.price) - float(best_bid.price)
                    mid = (float(best_ask.price) + float(best_bid.price)) / 2
                    print(f"Spread: ${spread:.4f}, Mid: ${mid:.4f}")
                    
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(debug_both_tokens())
