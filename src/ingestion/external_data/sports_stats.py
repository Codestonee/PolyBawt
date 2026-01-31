"""Sports statistics and ELO ratings for sports markets."""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import aiohttp

from .base_source import ExternalDataSource, DataSourceResult, DataSourceStatus
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EloRatings:
    """ELO ratings for a matchup."""
    team_a_elo: float
    team_b_elo: float
    
    # Adjustments
    home_advantage: float = 65.0  # Typical NBA/NFL home advantage
    team_a_rest_days: int = 0
    team_b_rest_days: int = 0
    team_a_injuries: list[str] = field(default_factory=list)
    team_b_injuries: list[str] = field(default_factory=list)
    
    def win_probability(self, is_team_a_home: bool = False) -> float:
        """
        Calculate win probability using ELO formula.
        
        P(A wins) = 1 / (1 + 10^((Rb-Ra)/400))
        """
        # Apply home advantage
        advantage = self.home_advantage if is_team_a_home else -self.home_advantage
        
        # Apply rest adjustment (fatigue)
        rest_adj = 20 * (self.team_b_rest_days - self.team_a_rest_days)
        
        effective_diff = (self.team_a_elo - self.team_b_elo) + advantage + rest_adj
        
        return 1.0 / (1.0 + 10.0 ** (-effective_diff / 400.0))


@dataclass
class BettingLine:
    """Vegas betting line."""
    sportsbook: str
    team_a_moneyline: int  # American odds, e.g., -150
    team_b_moneyline: int
    spread: float  # Team A spread
    over_under: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def implied_probability_team_a(self) -> float:
        """Convert moneyline to implied probability."""
        if self.team_a_moneyline < 0:
            return abs(self.team_a_moneyline) / (abs(self.team_a_moneyline) + 100)
        else:
            return 100 / (self.team_a_moneyline + 100)
    
    def implied_probability_team_b(self) -> float:
        """Convert moneyline to implied probability."""
        if self.team_b_moneyline < 0:
            return abs(self.team_b_moneyline) / (abs(self.team_b_moneyline) + 100)
        else:
            return 100 / (self.team_b_moneyline + 100)
    
    def vig_free_probability_team_a(self) -> float:
        """Remove vigorish to get true implied probability."""
        raw_a = self.implied_probability_team_a()
        raw_b = self.implied_probability_team_b()
        total = raw_a + raw_b
        return raw_a / total  # Normalize


@dataclass
class SportsData:
    """Complete sports data for a matchup."""
    matchup_id: str
    sport: str  # NBA, NFL, MLB, etc.
    team_a: str
    team_b: str
    game_time: datetime
    
    elo: EloRatings | None = None
    betting_lines: list[BettingLine] = field(default_factory=list)
    
    # Recent form
    team_a_last_10_wins: int = 0
    team_b_last_10_wins: int = 0
    
    # Head to head
    h2h_wins_a: int = 0
    h2h_wins_b: int = 0
    
    @property
    def vegas_consensus_prob(self) -> float | None:
        """Average vig-free probability across sportsbooks."""
        if not self.betting_lines:
            return None
        
        probs = [line.vig_free_probability_team_a() for line in self.betting_lines]
        return sum(probs) / len(probs)
    
    @property
    def elo_win_prob(self) -> float | None:
        """ELO-based win probability."""
        if self.elo is None:
            return None
        return self.elo.win_probability()
    
    def detect_edge(self) -> float:
        """
        Detect edge between ELO model and Vegas.
        
        Returns: Positive = model favors team A more than Vegas
        """
        elo_prob = self.elo_win_prob
        vegas_prob = self.vegas_consensus_prob
        
        if elo_prob is None or vegas_prob is None:
            return 0.0
        
        return elo_prob - vegas_prob


class SportsStatsSource(ExternalDataSource):
    """
    Fetches sports statistics and betting lines.
    
    Data sources:
    - FiveThirtyEight ELO ratings
    - Vegas betting lines (via APIs)
    - Injury reports
    
    Usage:
        source = SportsStatsSource()
        result = await source.fetch_cached(sport="NBA", team_a="LAL", team_b="BOS")
        
        if result.is_valid:
            data = result.data.get("sports_data")
            edge = data.detect_edge()
    """
    
    def __init__(self, api_key: str | None = None):
        super().__init__(name="SportsStats", api_key=api_key)
        self._session: aiohttp.ClientSession | None = None
        
        # 538 ELO endpoint (public)
        self.fivethirtyeight_url = "https://projects.fivethirtyeight.com/data.json"
        
        # Cached ELO data
        self._elo_cache: dict[str, dict[str, float]] = {}
        self._elo_cache_time: datetime | None = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    @property
    def update_frequency_seconds(self) -> int:
        return 900  # 15 minutes
    
    async def fetch(
        self,
        sport: str,
        team_a: str,
        team_b: str,
        game_date: str | None = None,
        **kwargs
    ) -> DataSourceResult:
        """
        Fetch sports data for a matchup.
        
        Args:
            sport: Sport code (NBA, NFL, MLB, etc.)
            team_a: Team A name/code
            team_b: Team B name/code
            game_date: Optional game date (YYYY-MM-DD)
            
        Returns:
            DataSourceResult with SportsData
        """
        start_time = datetime.now(timezone.utc)
        matchup_id = f"{sport}:{team_a}:{team_b}:{game_date or 'next'}"
        
        try:
            # Fetch ELO ratings
            elo = await self._fetch_elo(sport, team_a, team_b)
            
            # Fetch betting lines
            lines = await self._fetch_betting_lines(sport, team_a, team_b)
            
            # Build sports data
            sports_data = SportsData(
                matchup_id=matchup_id,
                sport=sport,
                team_a=team_a,
                team_b=team_b,
                game_time=self._parse_game_time(game_date),
                elo=elo,
                betting_lines=lines,
            )
            
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return DataSourceResult(
                data={"sports_data": sports_data},
                source_name=self.name,
                status=DataSourceStatus.HEALTHY,
                timestamp=start_time,
                latency_ms=latency_ms,
                confidence=0.8 if elo and lines else 0.5,
                sample_size=len(lines),
            )
            
        except Exception as e:
            logger.error("Sports stats fetch failed", error=str(e))
            return DataSourceResult(
                source_name=self.name,
                status=DataSourceStatus.DOWN,
                error_message=str(e),
            )
    
    async def _fetch_elo(self, sport: str, team_a: str, team_b: str) -> EloRatings | None:
        """Fetch ELO ratings from 538."""
        try:
            # Refresh cache if needed
            if self._should_refresh_elo():
                await self._refresh_elo_cache()
            
            # Look up teams
            team_a_elo = self._elo_cache.get(sport, {}).get(team_a, 1500)
            team_b_elo = self._elo_cache.get(sport, {}).get(team_b, 1500)
            
            return EloRatings(
                team_a_elo=team_a_elo,
                team_b_elo=team_b_elo,
            )
            
        except Exception as e:
            logger.warning("ELO fetch failed", error=str(e))
            return None
    
    def _should_refresh_elo(self) -> bool:
        """Check if ELO cache needs refresh."""
        if self._elo_cache_time is None:
            return True
        
        age = datetime.now(timezone.utc) - self._elo_cache_time
        return age > timedelta(hours=1)
    
    async def _refresh_elo_cache(self) -> None:
        """Refresh ELO ratings cache from 538."""
        try:
            session = await self._get_session()
            
            async with session.get(self.fivethirtyeight_url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Parse NBA ELO
                    if "nba" in data:
                        self._elo_cache["NBA"] = {
                            team["team"]: team["elo"]
                            for team in data["nba"].get("teams", [])
                        }
                    
                    # Parse NFL ELO
                    if "nfl" in data:
                        self._elo_cache["NFL"] = {
                            team["team"]: team["elo"]
                            for team in data["nfl"].get("teams", [])
                        }
                    
                    self._elo_cache_time = datetime.now(timezone.utc)
                    
        except Exception as e:
            logger.warning("ELO cache refresh failed", error=str(e))
    
    async def _fetch_betting_lines(
        self,
        sport: str,
        team_a: str,
        team_b: str
    ) -> list[BettingLine]:
        """Fetch betting lines from sportsbook APIs."""
        lines: list[BettingLine] = []
        
        # NOTE: In production, integrate with:
        # - OddsAPI (odds-api.com)
        # - Pinnacle API
        # - BetMGM/Sportsbook APIs
        
        # Mock implementation for structure
        logger.debug("Fetching betting lines", sport=sport, team_a=team_a, team_b=team_b)
        
        return lines
    
    def _parse_game_time(self, game_date: str | None) -> datetime:
        """Parse game date string to datetime."""
        if game_date:
            return datetime.strptime(game_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) + timedelta(days=1)
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Pre-instantiated instance
sports_stats = SportsStatsSource()
