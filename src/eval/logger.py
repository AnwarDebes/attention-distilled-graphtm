"""Experiment logging: JSONL per-epoch metrics with hardware/config metadata."""

import json
import os
import sys
import time
import platform
import subprocess
import hashlib


def get_hardware_info():
    """Collect hardware information for reproducibility."""
    gpu_name = ""
    cuda_version = ""
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5
        ).strip()
        cuda_version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5
        ).strip()
    except Exception:
        pass

    return {
        "gpu": gpu_name,
        "driver_version": cuda_version,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def get_git_hash():
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        return "unknown"


def hash_file(path):
    """SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class ExperimentLogger:
    """JSONL logger for experiment runs.

    Usage:
        logger = ExperimentLogger("experiments/my_exp/seed_42/log.jsonl")
        logger.log_config(config_dict)
        for epoch in range(n):
            logger.log_epoch(epoch, {"accuracy": acc, "loss": loss, "time": t})
        logger.log_summary(summary_dict)
    """

    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.f = open(log_path, "w")
        self.start_time = time.time()

    def _write(self, entry):
        entry["timestamp"] = time.time()
        self.f.write(json.dumps(entry) + "\n")
        self.f.flush()

    def log_config(self, config, dataset_hashes=None):
        """Log experiment configuration and hardware info."""
        self._write({
            "type": "config",
            "config": config,
            "hardware": get_hardware_info(),
            "git_hash": get_git_hash(),
            "dataset_hashes": dataset_hashes or {},
        })

    def log_epoch(self, epoch, metrics):
        """Log per-epoch metrics."""
        self._write({
            "type": "epoch",
            "epoch": epoch,
            **metrics,
        })

    def log_summary(self, summary):
        """Log final summary."""
        summary["total_time"] = time.time() - self.start_time
        self._write({
            "type": "summary",
            **summary,
        })

    def close(self):
        self.f.close()
