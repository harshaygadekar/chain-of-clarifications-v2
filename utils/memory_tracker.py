"""
Memory Tracker Module

Provides utilities for monitoring GPU and CPU memory usage during experiments.
Critical for ensuring the system runs within the 6GB VRAM constraint.
"""

import torch
import psutil
import os
from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryTracker:
    """
    Tracks memory usage throughout the execution.

    Monitors both GPU (if available) and system RAM.
    """

    def __init__(self, log_interval: int = 1):
        """
        Initialize the memory tracker.

        Args:
            log_interval: How often to log memory (every N calls)
        """
        self.log_interval = log_interval
        self.call_count = 0
        self.has_cuda = torch.cuda.is_available()
        self.measurements = []
        self.peak_gpu_memory = 0
        self.peak_ram_memory = 0

    def get_current_memory(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory statistics in MB
        """
        memory_info = {}

        # GPU memory
        if self.has_cuda:
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2

            # Update peak
            current_gpu = memory_info['gpu_allocated_mb']
            if current_gpu > self.peak_gpu_memory:
                self.peak_gpu_memory = current_gpu
        else:
            memory_info['gpu_allocated_mb'] = 0
            memory_info['gpu_reserved_mb'] = 0
            memory_info['gpu_max_allocated_mb'] = 0

        # System RAM
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / 1024**2
        memory_info['ram_mb'] = ram_mb

        # Update peak RAM
        if ram_mb > self.peak_ram_memory:
            self.peak_ram_memory = ram_mb

        # Total system memory
        virtual_mem = psutil.virtual_memory()
        memory_info['total_ram_mb'] = virtual_mem.total / 1024**2
        memory_info['available_ram_mb'] = virtual_mem.available / 1024**2
        memory_info['ram_percent'] = virtual_mem.percent

        # Timestamp
        memory_info['timestamp'] = datetime.now().isoformat()

        return memory_info

    def log_memory(self, label: str = ""):
        """
        Log current memory usage.

        Args:
            label: Optional label for this measurement
        """
        self.call_count += 1

        # Only log at intervals
        if self.call_count % self.log_interval != 0:
            return

        mem_info = self.get_current_memory()
        self.measurements.append({
            'label': label,
            **mem_info
        })

        # Log to console
        if self.has_cuda:
            logger.info(
                f"Memory [{label}]: "
                f"GPU={mem_info['gpu_allocated_mb']:.1f}MB, "
                f"RAM={mem_info['ram_mb']:.1f}MB"
            )
        else:
            logger.info(
                f"Memory [{label}]: "
                f"RAM={mem_info['ram_mb']:.1f}MB"
            )

    def check_memory_available(
        self,
        required_mb: float,
        memory_type: str = "gpu"
    ) -> bool:
        """
        Check if enough memory is available.

        Args:
            required_mb: Required memory in MB
            memory_type: "gpu" or "ram"

        Returns:
            True if enough memory available
        """
        mem_info = self.get_current_memory()

        if memory_type == "gpu":
            if not self.has_cuda:
                return False
            # Assume max GPU memory is 6GB for RTX 4050
            max_gpu_mb = 6144
            allocated = mem_info['gpu_allocated_mb']
            available = max_gpu_mb - allocated
            return available >= required_mb
        else:
            return mem_info['available_ram_mb'] >= required_mb

    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()
        self.peak_gpu_memory = 0
        self.peak_ram_memory = 0

    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak memory usage.

        Returns:
            Dictionary with peak memory statistics
        """
        return {
            'peak_gpu_mb': self.peak_gpu_memory,
            'peak_ram_mb': self.peak_ram_memory
        }

    def print_memory_summary(self):
        """Print a summary of memory usage."""
        peak = self.get_peak_memory()
        current = self.get_current_memory()

        print("\n" + "=" * 60)
        print("MEMORY USAGE SUMMARY")
        print("=" * 60)

        if self.has_cuda:
            print(f"\nGPU Memory:")
            print(f"  Current: {current['gpu_allocated_mb']:.1f} MB")
            print(f"  Peak:    {peak['peak_gpu_mb']:.1f} MB")
            print(f"  Reserved: {current['gpu_reserved_mb']:.1f} MB")
            print(f"  Limit:   6144.0 MB (RTX 4050)")
            print(f"  Usage:   {(peak['peak_gpu_mb']/6144)*100:.1f}%")
        else:
            print("\nGPU: Not available (using CPU)")

        print(f"\nSystem RAM:")
        print(f"  Current: {current['ram_mb']:.1f} MB")
        print(f"  Peak:    {peak['peak_ram_mb']:.1f} MB")
        print(f"  Total:   {current['total_ram_mb']:.1f} MB")
        print(f"  Usage:   {current['ram_percent']:.1f}%")

        print("=" * 60 + "\n")

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        if self.has_cuda:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def get_all_measurements(self) -> list:
        """
        Get all memory measurements.

        Returns:
            List of all measurements
        """
        return self.measurements


def estimate_model_memory(
    model_name: str,
    dtype: str = "float16"
) -> float:
    """
    Estimate memory required for a model.

    Args:
        model_name: Name of the model
        dtype: Data type (float32, float16)

    Returns:
        Estimated memory in MB
    """
    # Model parameter counts (approximate)
    param_counts = {
        "gpt2": 124e6,
        "gpt2-medium": 355e6,
        "gpt2-large": 774e6,
        "distilgpt2": 82e6,
    }

    # Get parameter count
    params = param_counts.get(model_name, 124e6)

    # Bytes per parameter
    if dtype == "float32":
        bytes_per_param = 4
    elif dtype == "float16":
        bytes_per_param = 2
    else:
        bytes_per_param = 4  # default

    # Estimate: params * bytes + 20% overhead for activations
    memory_bytes = params * bytes_per_param * 1.2
    memory_mb = memory_bytes / 1024**2

    return memory_mb


def get_gpu_info() -> Dict:
    """
    Get GPU information.

    Returns:
        Dictionary with GPU details
    """
    info = {
        'available': torch.cuda.is_available(),
        'device_count': 0,
        'device_name': None,
        'cuda_version': None,
    }

    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda

    return info
