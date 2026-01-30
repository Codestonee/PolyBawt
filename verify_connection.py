"""
Quick connection verification script for Polymarket.
Tests that credentials are correctly configured.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def main():
    print("=" * 60)
    print("[+] POLYMARKET CONNECTION TEST")
    print("=" * 60)
    
    # Check environment variables
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")
    
    print("\n[*] Environment Variables:")
    if private_key:
        # Show first/last 4 chars only
        masked = f"{private_key[:6]}...{private_key[-4:]}" if len(private_key) > 10 else "***"
        print(f"  [OK] POLYMARKET_PRIVATE_KEY: {masked}")
    else:
        print("  [FAIL] POLYMARKET_PRIVATE_KEY: NOT SET")
        sys.exit(1)
    
    if funder_address:
        print(f"  [OK] POLYMARKET_FUNDER_ADDRESS: {funder_address}")
    else:
        print("  [WARN] POLYMARKET_FUNDER_ADDRESS: NOT SET (using signer address)")
    
    # Try to initialize CLOB client
    print("\n[*] Initializing CLOB Client...")
    try:
        from py_clob_client.client import ClobClient
        
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,  # Polygon
            signature_type=0,  # EOA
            funder=funder_address,
        )
        
        # Derive API credentials
        print("  -> Deriving API credentials from private key...")
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        
        print(f"  [OK] API Key: {creds.api_key[:8]}...{creds.api_key[-4:]}")
        print(f"  [OK] API Secret: ***")
        print(f"  [OK] API Passphrase: ***")
        
    except Exception as e:
        print(f"  [FAIL] CLOB Client Error: {e}")
        sys.exit(1)
    
    # Try to fetch open orders (tests auth)
    print("\n[*] Testing API Authentication...")
    try:
        orders = client.get_orders()
        print(f"  [OK] Auth successful! Found {len(orders) if orders else 0} open orders.")
    except Exception as e:
        print(f"  [FAIL] Auth failed: {e}")
        print("\n  [WARN] This may be expected if you haven't traded before.")
        print("         Try making a small trade on polymarket.com first.")
    
    # Try to fetch a market (tests read access)
    print("\n[*] Testing Market Data Access...")
    try:
        # Get any order book to test read access
        # Use a known active market token
        book = client.get_order_book("21742633143463906290569050155826241533067272736897614950488156847949938836455")
        if book:
            print(f"  [OK] Market data access works!")
            if hasattr(book, 'bids') and book.bids:
                print(f"       Best bid: {book.bids[0].price if book.bids else 'N/A'}")
            if hasattr(book, 'asks') and book.asks:
                print(f"       Best ask: {book.asks[0].price if book.asks else 'N/A'}")
        else:
            print("  [WARN] Order book returned empty (market may be inactive)")
    except Exception as e:
        print(f"  [WARN] Market data fetch issue: {e}")
    
    print("\n" + "=" * 60)
    print("[OK] CONNECTION TEST COMPLETE")
    print("=" * 60)
    print("\nYour bot should be ready to run!")
    print("Start with: python -m src.main --dry-run")


if __name__ == "__main__":
    asyncio.run(main())
