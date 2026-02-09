"""
Monitoring Module for PolyBawt.

Provides:
- alerter: Multi-channel alerting (Telegram, Discord)
- dashboard: Real-time metrics aggregation
- health_monitor: System health checks and anomaly detection
"""

from src.monitoring.alerter import (
    Alerter,
    AlertConfig,
    AlertLevel,
    AlertChannel,
    Alert,
    init_alerter,
    get_alerter,
    alert,
    alert_info,
    alert_warning,
    alert_error,
    alert_critical,
)

from src.monitoring.dashboard import (
    DashboardProvider,
    RollingStats,
    TimeSeriesPoint,
    get_dashboard_provider,
)

from src.monitoring.health_monitor import (
    HealthMonitor,
    HealthConfig,
    HealthStatus,
    ComponentHealth,
    get_health_monitor,
)

__all__ = [
    # Alerter
    "Alerter",
    "AlertConfig",
    "AlertLevel",
    "AlertChannel",
    "Alert",
    "init_alerter",
    "get_alerter",
    "alert",
    "alert_info",
    "alert_warning",
    "alert_error",
    "alert_critical",
    # Dashboard
    "DashboardProvider",
    "RollingStats",
    "TimeSeriesPoint",
    "get_dashboard_provider",
    # Health Monitor
    "HealthMonitor",
    "HealthConfig",
    "HealthStatus",
    "ComponentHealth",
    "get_health_monitor",
]
