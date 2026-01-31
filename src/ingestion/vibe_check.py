"""
Vibe Check: Sentiment Analysis Engine for Crypto Markets.
Part of the "Grey Zone" toolkit.

Fetches tweets from "Alpha" accounts via Nitter (privacy front-end) to avoid API limits.
Analyzes sentiment to produce a Vibe Score (-1.0 to 1.0).
"""

import asyncio
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
import aiohttp
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

# The Alpha List: Accounts that move markets
TARGET_ACCOUNTS = [
    "tier10k",       # DB (News)
    "WatcherGuru",   # Breaking news
    "Tree_of_Alpha", # Fast trading news
    "Whale_Alert",   # Large movements
]

# Nitter instances to rotate (Grey Zone bypass)
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.cz",
    "https://nitter.privacydev.net",
    "https://nitter.projectsegfau.lt",
]

@dataclass
class VibeSignal:
    score: float  # -1.0 (Bearish) to 1.0 (Bullish)
    confidence: float # 0.0 to 1.0
    sources: list[str]
    timestamp: float

class VibeChecker:
    def __init__(self):
        self.session = None
        # Keyword-based heuristics (Fallback until LLM is hooked up)
        self.bullish_words = {"approved", "partnership", "launch", "bull", "buying", "support", "etf", "green"}
        self.bearish_words = {"banned", "hack", "exploit", "crash", "bear", "selling", "resistance", "red", "sec", "sued"}

    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
        return self.session

    async def fetch_tweets(self, username: str) -> list[str]:
        """Scrape tweets from Nitter instances with rotation."""
        session = await self._get_session()
        
        # Shuffle instances to distribute load
        instances = list(NITTER_INSTANCES)
        random.shuffle(instances)

        for base_url in instances:
            try:
                url = f"{base_url}/{username}/rss"
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Simple regex to extract titles (tweets) from RSS
                        tweets = re.findall(r'<title>(.*?)</title>', content)
                        # Filter out "Twitter / username" title
                        clean_tweets = [t for t in tweets if f" / {username}" not in t and "Twitter" not in t]
                        if clean_tweets:
                            logger.info(f"Got {len(clean_tweets)} tweets from {username} via {base_url}")
                            return clean_tweets[:5] # Last 5 tweets
            except Exception as e:
                logger.debug(f"Failed to fetch {username} from {base_url}: {e}")
                continue
        
        logger.warning(f"Could not fetch tweets for {username} from any instance.")
        return []

    def score_text(self, text: str) -> float:
        """
        Rudimentary sentiment scoring.
        TODO: Replace with Gemini Flash API call for 'Amazing' tier.
        """
        text = text.lower()
        score = 0.0
        
        for word in self.bullish_words:
            if word in text:
                score += 0.5
        
        for word in self.bearish_words:
            if word in text:
                score -= 0.5
                
        # Clamp between -1 and 1
        return max(-1.0, min(1.0, score))

    async def get_vibe(self) -> VibeSignal:
        """Run the full vibe check."""
        logger.info("Running Vibe Check...")
        total_score = 0.0
        total_tweets = 0
        sources = []

        tasks = [self.fetch_tweets(user) for user in TARGET_ACCOUNTS]
        results = await asyncio.gather(*tasks)

        for username, tweets in zip(TARGET_ACCOUNTS, results):
            if not tweets:
                continue
            
            user_score = 0.0
            for tweet in tweets:
                s = self.score_text(tweet)
                user_score += s
                if s != 0:
                    sources.append(f"@{username}: {tweet[:30]}... ({s})")
            
            # Average score for this user
            if tweets:
                avg_user_score = user_score / len(tweets)
                # Weighted impact? For now, equal weight.
                total_score += avg_user_score
                total_tweets += 1

        final_score = 0.0
        if total_tweets > 0:
            final_score = total_score / total_tweets
            final_score = max(-1.0, min(1.0, final_score))

        # Confidence based on how many sources we actually reached
        confidence = total_tweets / len(TARGET_ACCOUNTS)

        return VibeSignal(
            score=final_score,
            confidence=confidence,
            sources=sources,
            timestamp=datetime.now(timezone.utc).timestamp()
        )

    async def close(self):
        if self.session:
            await self.session.close()

# Quick test entry point
if __name__ == "__main__":
    async def main():
        vibe = VibeChecker()
        signal = await vibe.get_vibe()
        print(f"\n--- VIBE CHECK REPORT ---")
        print(f"Score: {signal.score:.2f} (-1.0 = Bear, 1.0 = Bull)")
        print(f"Confidence: {signal.confidence:.0%}")
        print("Sources:")
        for s in signal.sources:
            print(f" - {s}")
        await vibe.close()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
