"""
System Health Monitor for PolyBawt.

Continuously monitors system health and triggers alerts:
- Component health checks
- Anomaly detection
- Resource monitoring
- Scheduled reports

Usage:
    monitor = HealthMonitor()
    await monitor.start()
"""

from __future__ import annotations
import asyncio
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum

from src.infrastructure.logging import get_logger
from src.monitoring.alerter import get_alerter, AlertLevel

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component."""
    name: str
    status: HealthStatus
    last_check: float
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthConfig:
    """Configuration for health monitoring."""
    check_interval_seconds: float = 30.0
    report_interval_seconds: float = 86400.0  # Daily
    
    # Thresholds
    memory_warning_pct: float = 80.0
    memory_critical_pct: float = 95.0
    cpu_warning_pct: float = 80.0
    latency_warning_ms: float = 500.0
    stale_data_seconds: float = 60.0
    
    # Anomaly detection
    pnl_drop_alert_usd: float = 100.0
    drawdown_warning_pct: float = 0.05  # 5%


class HealthMonitor:
    """
    System health monitor with alerting.
    
    Runs periodic health checks and sends alerts
    when issues are detected.
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        
        self._components: Dict[str, ComponentHealth] = {}
        self._check_tasks: List[asyncio.Task] = []
        self._running = False
        
        # Tracking
        self._last_pnl = 0.0
        self._peak_pnl = 0.0
        self._last_report_time = 0.0
        self._alerts_sent = 0
        
        # Custom health checks
        self._custom_checks: List[Callable[[], Awaitable[ComponentHealth]]] = []
    
    def register_check(self, check: Callable[[], Awaitable[ComponentHealth]]):
        """Register a custom health check."""
        self._custom_checks.append(check)
    
    async def check_memory(self) -> ComponentHealth:
        """Check system memory usage."""
        try:
            mem = psutil.virtual_memory()
            pct = mem.percent
            
            if pct >= self.config.memory_critical_pct:
                status = HealthStatus.UNHEALTHY
            elif pct >= self.config.memory_warning_pct:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="memory",
                status=status,
                last_check=time.time(),
                message=f"{pct:.1f}% used",
                metrics={"used_pct": pct, "available_gb": mem.available / 1e9},
            )
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                last_check=time.time(),
                message=str(e),
            )
    
    async def check_cpu(self) -> ComponentHealth:
        """Check CPU usage."""
        try:
            pct = psutil.cpu_percent(interval=0.1)
            
            if pct >= self.config.cpu_warning_pct:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="cpu",
                status=status,
                last_check=time.time(),
                message=f"{pct:.1f}% used",
                metrics={"used_pct": pct},
            )
        except Exception as e:
            return ComponentHealth(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                last_check=time.time(),
                message=str(e),
            )
    
    async def check_event_bus(self) -> ComponentHealth:
        """Check event bus health."""
        try:
            from src.infrastructure.events import event_bus
            
            stats = event_bus.stats
            dropped = stats.get("dropped", 0)
            
            if dropped > 100:
                status = HealthStatus.DEGRADED
            elif dropped > 0:
                status = HealthStatus.HEALTHY  # Some drops acceptable
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="event_bus",
                status=status,
                last_check=time.time(),
                message=f"{dropped} dropped events",
                metrics=stats,
            )
        except Exception as e:
            return ComponentHealth(
                name="event_bus",
                status=HealthStatus.UNKNOWN,
                last_check=time.time(),
                message=str(e),
            )
    
    async def check_circuit_breakers(self) -> ComponentHealth:
        """Check circuit breaker states."""
        try:
            from src.infrastructure.circuit_breaker import _global_registry
            
            if not _global_registry or not _global_registry._breakers:
                return ComponentHealth(
                    name="circuit_breakers",
                    status=HealthStatus.HEALTHY,
                    last_check=time.time(),
                    message="No breakers registered",
                )
            
            open_breakers = [
                name for name, b in _global_registry._breakers.items()
                if b.is_open
            ]
            
            if open_breakers:
                status = HealthStatus.DEGRADED
                message = f"Open: {', '.join(open_breakers)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"{len(_global_registry._breakers)} breakers OK"
            
            return ComponentHealth(
                name="circuit_breakers",
                status=status,
                last_check=time.time(),
                message=message,
            )
        except ImportError:
            return ComponentHealth(
                name="circuit_breakers",
                status=HealthStatus.UNKNOWN,
                last_check=time.time(),
                message="Module not available",
            )
    
    async def check_kill_switch(self) -> ComponentHealth:
        """Check kill switch state."""
        try:
            from src.infrastructure.kill_switch import _global_kill_switch
            
            if _global_kill_switch and _global_kill_switch.is_killed:
                return ComponentHealth(
                    name="kill_switch",
                    status=HealthStatus.UNHEALTHY,
                    last_check=time.time(),
                    message=f"KILLED: {_global_kill_switch.kill_reason}",
                )
            
            return ComponentHealth(
                name="kill_switch",
                status=HealthStatus.HEALTHY,
                last_check=time.time(),
                message="Not triggered",
            )
        except ImportError:
            return ComponentHealth(
                name="kill_switch",
                status=HealthStatus.UNKNOWN,
                last_check=time.time(),
                message="Module not available",
            )
    
    async def run_all_checks(self) -> Dict[str, ComponentHealth]:
        """Run all health checks."""
        checks = [
            self.check_memory(),
            self.check_cpu(),
            self.check_event_bus(),
            self.check_circuit_breakers(),
            self.check_kill_switch(),
        ]
        
        # Add custom checks
        checks.extend(c() for c in self._custom_checks)
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        components = {}
        for result in results:
            if isinstance(result, ComponentHealth):
                components[result.name] = result
                self._components[result.name] = result
            elif isinstance(result, Exception):
                logger.error(f"[HealthMonitor] Check failed: {result}")
        
        return components
    
    async def _check_loop(self):
        """Periodic health check loop."""
        while self._running:
            try:
                components = await self.run_all_checks()
                
                # Check for issues and alert
                for name, health in components.items():
                    if health.status == HealthStatus.UNHEALTHY:
                        alerter = get_alerter()
                        await alerter.send_alert(
                            AlertLevel.ERROR,
                            f"Component Unhealthy: {name}",
                            health.message,
                            {"component": name, **health.metrics},
                        )
                        self._alerts_sent += 1
                    elif health.status == HealthStatus.DEGRADED:
                        alerter = get_alerter()
                        await alerter.send_alert(
                            AlertLevel.WARNING,
                            f"Component Degraded: {name}",
                            health.message,
                            {"component": name, **health.metrics},
                        )
                        self._alerts_sent += 1
                
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[HealthMonitor] Check loop error: {e}")
                await asyncio.sleep(5)
    
    async def _report_loop(self):
        """Periodic summary report loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.report_interval_seconds)
                await self.send_daily_report()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[HealthMonitor] Report loop error: {e}")
    
    async def send_daily_report(self):
        """Send daily summary report."""
        try:
            from src.api.state import get_state
            from src.monitoring.dashboard import get_dashboard_provider
            
            state = get_state()
            dashboard = get_dashboard_provider()
            
            data = dashboard.get_dashboard_data()
            pnl = data.get("pnl", {})
            orders = data.get("orders", {})
            uptime = data.get("uptime", {})
            
            alerter = get_alerter()
            await alerter.send_alert(
                AlertLevel.INFO,
                "Daily Summary Report",
                f"Uptime: {uptime.get('uptime_human', 'N/A')}",
                {
                    "total_pnl": pnl.get("total_pnl", 0),
                    "orders_placed": orders.get("orders_placed", 0),
                    "fill_rate": f"{orders.get('fill_rate_pct', 0)}%",
                    "alerts_sent": self._alerts_sent,
                },
                force=True,  # Always send daily report
            )
            
            self._last_report_time = time.time()
            
        except Exception as e:
            logger.error(f"[HealthMonitor] Daily report error: {e}")
    
    async def check_pnl_anomaly(self, current_pnl: float):
        """Check for PnL anomalies and alert."""
        # Check sudden drop
        pnl_change = current_pnl - self._last_pnl
        if pnl_change < -self.config.pnl_drop_alert_usd:
            alerter = get_alerter()
            await alerter.send_alert(
                AlertLevel.WARNING,
                "Sudden PnL Drop",
                f"PnL dropped ${abs(pnl_change):.2f} in last period",
                {"current_pnl": current_pnl, "change": pnl_change},
            )
        
        # Check drawdown
        self._peak_pnl = max(self._peak_pnl, current_pnl)
        drawdown = (self._peak_pnl - current_pnl) / max(1, self._peak_pnl)
        
        if drawdown >= self.config.drawdown_warning_pct:
            alerter = get_alerter()
            await alerter.send_alert(
                AlertLevel.WARNING,
                "Drawdown Warning",
                f"Drawdown at {drawdown*100:.1f}%",
                {"peak": self._peak_pnl, "current": current_pnl},
            )
        
        self._last_pnl = current_pnl
    
    async def start(self):
        """Start the health monitor."""
        self._running = True
        self._check_tasks = [
            asyncio.create_task(self._check_loop()),
            asyncio.create_task(self._report_loop()),
        ]
        logger.info("[HealthMonitor] Started")
    
    async def stop(self):
        """Stop the health monitor."""
        self._running = False
        for task in self._check_tasks:
            task.cancel()
        await asyncio.gather(*self._check_tasks, return_exceptions=True)
        logger.info("[HealthMonitor] Stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "running": self._running,
            "components": {
                name: {
                    "status": c.status.value,
                    "message": c.message,
                    "last_check": c.last_check,
                }
                for name, c in self._components.items()
            },
            "alerts_sent": self._alerts_sent,
            "overall": self._get_overall_status().value,
        }
    
    def _get_overall_status(self) -> HealthStatus:
        """Get overall system health."""
        if not self._components:
            return HealthStatus.UNKNOWN
        
        statuses = [c.status for c in self._components.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if any(s == HealthStatus.UNKNOWN for s in statuses):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


# Global instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
