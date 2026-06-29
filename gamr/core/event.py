"""
gamr/core/event.py
==================
Event-driven architecture for GAMR.

Instead of polling or blocking, the runtime reacts to events.
Every action produces events. Events trigger scheduler decisions.
Events cascade — a KERNEL_FINISHED event may trigger a PREFETCH_TRIGGER,
which triggers an SSD_READ_START, which eventually produces SSD_READ_COMPLETE.

This is the cleanest possible design for an async streaming runtime.
No polling loops. No fixed timers. Pure reactive flow.

Design philosophy:
  "Don't ask if something is ready. React when it becomes ready."

Similar to:
  - Node.js event loop (but for tensor memory)
  - Linux kernel interrupt handling
  - Reactive Streams (but synchronous where needed for GPU ordering)
"""

from __future__ import annotations

import time
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Callable, Dict, List


class EventType(Enum):
    """All event types in the GAMR runtime."""

    # ─── I/O Events ───────────────────────────────────────────────
    SSD_READ_START      = auto()   # SSD read has been dispatched
    SSD_READ_COMPLETE   = auto()   # Block is now in RAM

    # ─── Transfer Events ──────────────────────────────────────────
    RAM_TO_VRAM_START    = auto()  # DMA transfer to GPU started
    RAM_TO_VRAM_COMPLETE = auto()  # Block is now in VRAM

    # ─── Compute Events ───────────────────────────────────────────
    KERNEL_LAUNCHED     = auto()   # GPU kernel has been submitted
    KERNEL_FINISHED     = auto()   # GPU kernel completed (synchronous point)

    # ─── GPU State Events ─────────────────────────────────────────
    GPU_IDLE            = auto()   # GPU finished a block and has nothing next
    GPU_STALLED         = auto()   # GPU is waiting for a block that isn't ready

    # ─── Memory Pressure Events ───────────────────────────────────
    RAM_PRESSURE_HIGH   = auto()   # RAM usage above threshold
    RAM_PRESSURE_NORMAL = auto()   # RAM usage back to normal
    VRAM_PRESSURE_HIGH  = auto()   # VRAM usage above threshold
    VRAM_PRESSURE_NORMAL = auto()  # VRAM pressure relieved

    # ─── Storage Health Events ────────────────────────────────────
    SSD_THROUGHPUT_DROP  = auto()  # Measured SSD speed dropped significantly
    SSD_THROUGHPUT_SPIKE = auto()  # SSD speed recovered

    # ─── Scheduler Control Events ─────────────────────────────────
    PREFETCH_TRIGGER     = auto()  # Instruct storage scheduler to prefetch
    BLOCK_EVICT_REQUEST  = auto()  # Request eviction of a specific block
    BLOCK_EVICT_COMPLETE = auto()  # Eviction finished
    PIPELINE_STALL       = auto()  # The whole pipeline has stalled

    # ─── Profiler Events ──────────────────────────────────────────
    STATS_SNAPSHOT       = auto()  # Periodic stats collection tick
    COST_MODEL_UPDATE    = auto()  # Cost model parameters should be refreshed

    # ─── Lifecycle Events ─────────────────────────────────────────
    RUNTIME_START        = auto()
    RUNTIME_STOP         = auto()
    EXECUTION_COMPLETE   = auto()


@dataclass
class GAMREvent:
    """
    A single event in the GAMR event stream.

    Every significant action in the runtime produces an event.
    Events carry a payload (block, timing data, stats, etc.) and
    a timestamp for performance analysis.
    """
    type:      EventType
    timestamp: float         = field(default_factory=time.perf_counter)
    payload:   Any           = None    # Block, stats dict, error, etc.
    source:    str           = ""      # Which component fired this event
    priority:  int           = 0       # Higher = processed first (for GPU_IDLE etc.)

    def __lt__(self, other: "GAMREvent") -> bool:
        """For priority queue ordering: higher priority first, then by timestamp."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority = earlier
        return self.timestamp < other.timestamp

    def __repr__(self) -> str:
        return f"GAMREvent({self.type.name}, src={self.source!r}, t={self.timestamp:.4f})"


class EventQueue:
    """
    Thread-safe event queue with priority support.

    The central nervous system of GAMR. All components (schedulers,
    profiler, pipeline threads) publish events here.
    The Dispatcher reads events and routes them to the right handler.

    Usage:
        eq = EventQueue()

        # Producer (any thread)
        eq.publish(GAMREvent(EventType.SSD_READ_COMPLETE, payload=block))

        # Consumer (dispatcher thread)
        event = eq.get(timeout=1.0)
    """

    def __init__(self, maxsize: int = 10_000):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=maxsize)
        self._handlers: Dict[EventType, List[Callable[[GAMREvent], None]]] = {}
        self._lock = threading.Lock()
        self._stats = {
            "total_published": 0,
            "total_consumed": 0,
            "peak_depth": 0,
        }

    def publish(self, event: GAMREvent) -> None:
        """
        Publish an event. Non-blocking (raises queue.Full if overloaded).
        Called from any thread.
        """
        self._queue.put_nowait(event)
        with self._lock:
            self._stats["total_published"] += 1
            depth = self._queue.qsize()
            if depth > self._stats["peak_depth"]:
                self._stats["peak_depth"] = depth

    def get(self, timeout: Optional[float] = None) -> Optional[GAMREvent]:
        """
        Consume the next event. Blocks until an event is available or timeout.
        Returns None on timeout.
        """
        try:
            event = self._queue.get(timeout=timeout)
            with self._lock:
                self._stats["total_consumed"] += 1
            return event
        except queue.Empty:
            return None

    def subscribe(self, event_type: EventType, handler: Callable[[GAMREvent], None]) -> None:
        """
        Register a handler for a specific event type.
        The dispatcher calls all registered handlers when an event arrives.
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def dispatch_one(self, timeout: float = 0.01) -> bool:
        """
        Get the next event and call all registered handlers.
        Returns True if an event was processed, False on timeout.
        """
        event = self.get(timeout=timeout)
        if event is None:
            return False

        with self._lock:
            handlers = list(self._handlers.get(event.type, []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Never let a handler crash the event loop
                print(f"[EventQueue] Handler error for {event.type.name}: {e}")

        return True

    def dispatch_loop(self, stop_event: threading.Event) -> None:
        """
        Run the dispatch loop until stop_event is set.
        Intended to run in its own thread.
        """
        while not stop_event.is_set():
            self.dispatch_one(timeout=0.005)

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def __repr__(self) -> str:
        return f"EventQueue(depth={self.depth}, published={self._stats['total_published']})"
