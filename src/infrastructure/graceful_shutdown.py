"""
Graceful Shutdown Manager for PolyBawt.

Handles clean shutdown with:
- Position cleanup
- Order cancellation
- State persistence
- Resource cleanup

Usage:
    shutdown = GracefulShutdown()
    shutdown.register_cleanup(cleanup_orders)
    
    # In main:
    await shutdown.wait_for_signal()
"""

from __future__ import annotations
import asyncio
import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Any, List, Awaitable, Dict
from enum import Enum

from src.infrastructure.logging import get_logger
from src.infrastructure.kill_switch import get_kill_switch

logger = get_logger(__name__)


class ShutdownPhase(Enum):
    """Phases of graceful shutdown."""
    RUNNING = "running"
    STOPPING = "stopping"
    CLEANUP = "cleanup"
    COMPLETE = "complete"


@dataclass
class CleanupTask:
    """A registered cleanup task."""
    name: str
    handler: Callable[[], Awaitable[None]]
    priority: int  # Lower = runs first
    timeout_seconds: float


class GracefulShutdown:
    """
    Graceful shutdown manager.
    
    Ensures clean shutdown by:
    1. Catching shutdown signals
    2. Triggering kill switch
    3. Running cleanup tasks in order
    4. Persisting state
    """
    
    def __init__(
        self,
        shutdown_timeout_seconds: float = 30.0,
    ):
        self.shutdown_timeout = shutdown_timeout_seconds
        
        self._phase = ShutdownPhase.RUNNING
        self._cleanup_tasks: List[CleanupTask] = []
        self._shutdown_event = asyncio.Event()
        self._shutdown_start_time = 0.0
        self._shutdown_reason = ""
        
        self._signal_handlers_installed = False
    
    @property
    def is_shutting_down(self) -> bool:
        return self._phase != ShutdownPhase.RUNNING
    
    @property
    def phase(self) -> ShutdownPhase:
        return self._phase
    
    def register_cleanup(
        self,
        name: str,
        handler: Callable[[], Awaitable[None]],
        priority: int = 50,
        timeout_seconds: float = 10.0,
    ):
        """
        Register a cleanup task.
        
        Args:
            name: Name for logging
            handler: Async cleanup function
            priority: Execution order (lower = first, default 50)
            timeout_seconds: Max time for this task
        """
        self._cleanup_tasks.append(CleanupTask(
            name=name,
            handler=handler,
            priority=priority,
            timeout_seconds=timeout_seconds,
        ))
        logger.debug(f"[Shutdown] Registered cleanup: {name} (priority={priority})")
    
    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return
        
        # Windows doesn't support all signals
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._handle_signal(s)),
                )
        else:
            # Windows: use signal.signal for SIGINT
            signal.signal(signal.SIGINT, self._sync_signal_handler)
        
        self._signal_handlers_installed = True
        logger.info("[Shutdown] Signal handlers installed")
    
    def _sync_signal_handler(self, signum, frame):
        """Synchronous signal handler for Windows."""
        logger.warning(f"[Shutdown] Received signal {signum}")
        asyncio.create_task(self.initiate_shutdown(f"signal_{signum}"))
    
    async def _handle_signal(self, sig: signal.Signals):
        """Handle shutdown signal."""
        logger.warning(f"[Shutdown] Received {sig.name}")
        await self.initiate_shutdown(f"signal_{sig.name}")
    
    async def initiate_shutdown(self, reason: str = "requested"):
        """
        Start the shutdown sequence.
        
        Args:
            reason: Reason for shutdown
        """
        if self._phase != ShutdownPhase.RUNNING:
            logger.warning("[Shutdown] Already shutting down")
            return
        
        self._phase = ShutdownPhase.STOPPING
        self._shutdown_start_time = time.time()
        self._shutdown_reason = reason
        
        logger.warning(f"[Shutdown] Initiating graceful shutdown: {reason}")
        
        # Trigger kill switch
        kill_switch = get_kill_switch()
        if not kill_switch.is_killed:
            await kill_switch.trigger("shutdown", f"Graceful shutdown: {reason}")
        
        # Run cleanup
        self._phase = ShutdownPhase.CLEANUP
        await self._run_cleanup()
        
        self._phase = ShutdownPhase.COMPLETE
        self._shutdown_event.set()
        
        elapsed = time.time() - self._shutdown_start_time
        logger.info(f"[Shutdown] Complete in {elapsed:.2f}s")
    
    async def _run_cleanup(self):
        """Run all registered cleanup tasks."""
        # Sort by priority
        sorted_tasks = sorted(self._cleanup_tasks, key=lambda t: t.priority)
        
        for task in sorted_tasks:
            logger.info(f"[Shutdown] Running cleanup: {task.name}")
            
            try:
                await asyncio.wait_for(
                    task.handler(),
                    timeout=task.timeout_seconds,
                )
                logger.info(f"[Shutdown] Cleanup complete: {task.name}")
            except asyncio.TimeoutError:
                logger.error(
                    f"[Shutdown] Cleanup timeout: {task.name} "
                    f"(>{task.timeout_seconds}s)"
                )
            except Exception as e:
                logger.error(f"[Shutdown] Cleanup failed: {task.name}: {e}")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to complete."""
        await self._shutdown_event.wait()
    
    async def wait_with_timeout(self) -> bool:
        """
        Wait for shutdown with timeout.
        
        Returns:
            True if shutdown completed normally
        """
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=self.shutdown_timeout,
            )
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"[Shutdown] Forced exit after {self.shutdown_timeout}s timeout"
            )
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get shutdown status."""
        return {
            "phase": self._phase.value,
            "is_shutting_down": self.is_shutting_down,
            "reason": self._shutdown_reason,
            "cleanup_tasks_registered": len(self._cleanup_tasks),
            "elapsed_seconds": (
                round(time.time() - self._shutdown_start_time, 2)
                if self._shutdown_start_time > 0 else 0
            ),
        }


# Global shutdown manager
_global_shutdown: Optional[GracefulShutdown] = None


def get_shutdown_manager() -> GracefulShutdown:
    """Get the global shutdown manager."""
    global _global_shutdown
    if _global_shutdown is None:
        _global_shutdown = GracefulShutdown()
    return _global_shutdown


def register_cleanup(
    name: str,
    handler: Callable[[], Awaitable[None]],
    priority: int = 50,
):
    """Register a cleanup task with the global shutdown manager."""
    get_shutdown_manager().register_cleanup(name, handler, priority)


async def shutdown(reason: str = "requested"):
    """Initiate shutdown on the global manager."""
    await get_shutdown_manager().initiate_shutdown(reason)
