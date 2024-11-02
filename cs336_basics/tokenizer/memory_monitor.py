import gc
import inspect
import mmap
import os
import resource
import sys
import threading
import time
import traceback
import tracemalloc
import weakref
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import psutil


class ComponentTracker:
    """Tracks memory usage for specific components or code sections."""

    def __init__(self):
        self.components = defaultdict(list)
        self.current_snapshot = None
        self.snapshots = {}
        self.object_refs = defaultdict(list)

    @contextmanager
    def track(self, component_name: str):
        """Context manager to track memory usage of a specific component."""
        # Take snapshot before
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        objects_before = len(gc.get_objects())
        process = psutil.Process()
        mem_before = process.memory_info()

        try:
            # Track all new objects created within this context
            gc.collect()  # Clean up before tracking
            objects_start = set(id(o) for o in gc.get_objects())

            yield

            # Find new objects
            gc.collect()
            objects_end = set(id(o) for o in gc.get_objects())
            new_objects = objects_end - objects_start

            # Store weak references to new objects
            for obj_id in new_objects:
                try:
                    obj = next(o for o in gc.get_objects() if id(o) == obj_id)
                    self.object_refs[component_name].append(weakref.ref(obj))
                except (StopIteration, TypeError):
                    continue

        finally:
            # Take snapshot after
            snapshot_after = tracemalloc.take_snapshot()
            objects_after = len(gc.get_objects())
            mem_after = process.memory_info()

            # Calculate memory changes
            memory_diff = {
                "rss": (mem_after.rss - mem_before.rss) / 1024 / 1024,  # MB
                "vms": (mem_after.vms - mem_before.vms) / 1024 / 1024,  # MB
                "objects": objects_after - objects_before,
                "timestamp": datetime.now(),
                "stack_trace": "".join(traceback.format_stack()[:-1]),
            }

            # Store snapshots for detailed analysis
            self.snapshots[component_name] = {
                "before": snapshot_before,
                "after": snapshot_after,
            }

            self.components[component_name].append(memory_diff)
            tracemalloc.stop()

    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """Get memory statistics for each tracked component."""
        stats = {}
        for component, measurements in self.components.items():
            if measurements:
                # Calculate basic statistics
                rss_values = [m["rss"] for m in measurements]
                vms_values = [m["vms"] for m in measurements]
                object_counts = [m["objects"] for m in measurements]

                stats[component] = {
                    "total_rss": sum(rss_values),
                    "peak_rss": max(rss_values),
                    "avg_rss": sum(rss_values) / len(rss_values),
                    "total_vms": sum(vms_values),
                    "peak_vms": max(vms_values),
                    "total_objects": sum(object_counts),
                    "calls": len(measurements),
                }

                # Add current live objects
                live_objects = sum(
                    1
                    for refs in self.object_refs[component]
                    for ref in [refs()]
                    if ref is not None
                )
                stats[component]["current_live_objects"] = live_objects

                # Calculate memory composition
                if component in self.snapshots:
                    stats[component]["memory_composition"] = (
                        self._analyze_memory_composition(
                            self.snapshots[component]["before"],
                            self.snapshots[component]["after"],
                        )
                    )

        return stats

    def _analyze_memory_composition(
        self, snapshot_before, snapshot_after
    ) -> Dict[str, float]:
        """Analyze what types of objects are contributing to memory usage."""
        statistics = snapshot_after.compare_to(snapshot_before, "traceback")

        composition = defaultdict(float)
        for stat in statistics[:10]:  # Top 10 memory blocks
            frame = stat.traceback[-1]
            filename = os.path.basename(frame.filename)
            line = frame.lineno
            size_mb = stat.size_diff / 1024 / 1024  # Convert to MB

            # Try to get the variable name or object type
            try:
                with open(frame.filename, "r") as f:
                    lines = f.readlines()
                    code_line = lines[line - 1].strip()
                    composition[f"{filename}:{line} ({code_line})"] = size_mb
            except:
                composition[f"{filename}:{line}"] = size_mb

        return dict(composition)


class EnhancedMemoryTracker:
    def __init__(self, snapshot_interval: int = 5):
        self.snapshot_interval = snapshot_interval
        self.tracking = False
        self.start_time = None
        self.process = psutil.Process()
        self.measurements = []
        self.component_tracker = ComponentTracker()

    # [Previous EnhancedMemoryTracker methods remain the same...]

    def start(self):
        """Start comprehensive memory tracking."""
        self.tracking = True
        self.start_time = datetime.now()

        # Start tracemalloc for Python object tracking
        tracemalloc.start()

        # Start background monitoring
        self.monitor_thread = threading.Thread(target=self._background_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        """Stop tracking and display results."""
        self.tracking = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()

        self._print_memory_analysis()
        tracemalloc.stop()

    def _get_process_memory_info(self) -> Dict[str, float]:
        """Get detailed memory information from multiple sources."""

        # Get memory info from psutil
        mem_info = self.process.memory_info()

        # Get memory info from resource module
        rusage = resource.getrusage(resource.RUSAGE_SELF)

        # Get specific memory maps info
        maps_info = self._get_memory_maps_info()

        # Get garbage collector statistics
        gc_count = gc.get_count()
        gc_objects = len(gc.get_objects())

        # Combine all memory information
        memory_info = {
            # RSS (Resident Set Size) - Total memory actually in RAM
            "rss": mem_info.rss / 1024 / 1024,
            # VMS (Virtual Memory Size) - Total virtual memory
            "vms": mem_info.vms / 1024 / 1024,
            # Shared memory
            "shared": mem_info.shared / 1024 / 1024
            if hasattr(mem_info, "shared")
            else 0,
            # Memory maps (anonymous, heap, stack)
            "anonymous": maps_info["anonymous"] / 1024 / 1024,
            "heap": maps_info["heap"] / 1024 / 1024,
            "stack": maps_info["stack"] / 1024 / 1024,
            # Maximum resident set size from resource module
            "max_rss_resource": rusage.ru_maxrss / 1024,  # Already in KB, convert to MB
            # Garbage collector info
            "gc_objects": gc_objects,
            "gc_generations": gc_count,
            # Memory from mmap'd files
            "mmap": maps_info["mmap"] / 1024 / 1024,
            # System-wide memory info
            "system_used": psutil.virtual_memory().used / 1024 / 1024,
            "system_available": psutil.virtual_memory().available / 1024 / 1024,
        }

        # Add numpy array memory if numpy is being used
        if "numpy" in sys.modules:
            memory_info["numpy_arrays"] = self._get_numpy_memory() / 1024 / 1024

        return memory_info

    def _get_memory_maps_info(self) -> Dict[str, int]:
        """Get detailed memory mapping information."""
        maps_info = {"anonymous": 0, "heap": 0, "stack": 0, "mmap": 0}

        try:
            # On Linux systems, read /proc/self/smaps
            if os.path.exists("/proc/self/smaps"):
                with open("/proc/self/smaps", "r") as f:
                    current_type = None
                    for line in f:
                        if "heap" in line.lower():
                            current_type = "heap"
                        elif "stack" in line.lower():
                            current_type = "stack"
                        elif "anonymous" in line.lower():
                            current_type = "anonymous"
                        elif line.startswith("Size:"):
                            size = int(line.split()[1])  # Size in kB
                            if current_type:
                                maps_info[current_type] += (
                                    size * 1024
                                )  # Convert to bytes

            # Get mmap'd file sizes
            for m in self.process.memory_maps():
                if m.path and m.path != "[heap]" and m.path != "[stack]":
                    maps_info["mmap"] += m.rss

        except (PermissionError, ProcessLookupError, FileNotFoundError):
            # Fallback to basic memory info if detailed info is not available
            mem = self.process.memory_info()
            maps_info["anonymous"] = mem.rss

        return maps_info

    def _get_numpy_memory(self) -> int:
        """Calculate total memory used by NumPy arrays."""
        total = 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, np.ndarray):
                    total += obj.nbytes
            except Exception:
                continue
        return total

    def _background_monitor(self):
        """Continuously monitor memory usage in background."""
        while self.tracking:
            mem_info = self._get_process_memory_info()
            self.measurements.append(
                {
                    "timestamp": (datetime.now() - self.start_time).total_seconds(),
                    "memory": mem_info,
                }
            )
            time.sleep(self.snapshot_interval)

    def track_component(self, component_name: str):
        """Get a context manager to track a specific component."""
        return self.component_tracker.track(component_name)

    def _print_memory_analysis(self):
        """Print comprehensive memory analysis including component-specific information."""
        # [Previous memory analysis code remains...]

        print("\n=== Component-Specific Memory Analysis ===")
        stats = self.component_tracker.get_component_stats()

        for component, data in stats.items():
            print(f"\nComponent: {component}")
            print(f"Total RSS Growth:          {data['total_rss']:>10.2f} MB")
            print(f"Peak RSS:                  {data['peak_rss']:>10.2f} MB")
            print(f"Average RSS:               {data['avg_rss']:>10.2f} MB")
            print(f"Total Objects Created:     {data['total_objects']:>10}")
            print(f"Currently Live Objects:    {data['current_live_objects']:>10}")
            print(f"Number of Calls:           {data['calls']:>10}")

            if "memory_composition" in data:
                print("\nMemory Composition:")
                for location, size in data["memory_composition"].items():
                    print(f"  {location}: {size:.2f} MB")


def track_memory(component_name: str = None):
    """Decorator to track memory usage of a function as a component."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            name = component_name or func.__name__
            with memory_tracker.component_tracker.track(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Create a global memory tracker instance
memory_tracker = EnhancedMemoryTracker()
