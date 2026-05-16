"""
Metrics collection with pluggable backends: Comet-ML, local JSON, stdout, or any combination.

Config section (in any YAML):

    logging:
      stdout: true          # always print a summary line to stdout
      comet: true           # send to Comet-ML (needs comet: block or COMET_API_KEY)
      json: true            # write metrics to outputs/<name>/metrics.jsonl
      json_path: null       # override path; if null, auto-derived from experiment.name

Usage:
    logger = MetricsLogger.from_config(config)
    logger.log_step(step, {"loss": 0.5, "reward/train_mean": 0.3})
    logger.finish()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


class MetricsLogger:
    """
    Fan-out logger that writes to any combination of:
      - stdout
      - Comet-ML experiment
      - local JSONL file  (one JSON object per step, newline-delimited)
    """

    def __init__(
        self,
        *,
        use_stdout: bool = True,
        experiment=None,
        json_path: Optional[Path] = None,
        prefix: str = "",
    ):
        self._use_stdout  = use_stdout
        self._experiment  = experiment
        self._json_path   = json_path
        self._prefix      = prefix
        self._json_handle = None

        if self._json_path is not None:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
            self._json_handle = open(self._json_path, "a", buffering=1)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "MetricsLogger":
        """
        Build a MetricsLogger from a config dict.

        Reads two top-level sections:
          logging:  controls which backends are active
          comet:    Comet-ML credentials (only used when logging.comet == true)

        Defaults when `logging` section is absent: stdout=true, comet=false, json=false.
        """
        log_cfg  = config.get("logging", {})
        use_stdout = bool(log_cfg.get("stdout", True))
        use_comet  = bool(log_cfg.get("comet",  False))
        use_json   = bool(log_cfg.get("json",   False))

        # ---- Comet-ML ----
        experiment = None
        if use_comet:
            experiment = _init_comet(config)

        # ---- JSON path ----
        json_path: Optional[Path] = None
        if use_json:
            override = log_cfg.get("json_path")
            if override:
                json_path = Path(override)
            else:
                exp_name = config.get("experiment", {}).get("name", "run")
                json_path = Path("outputs") / exp_name / "metrics.jsonl"

        logger = cls(
            use_stdout=use_stdout,
            experiment=experiment,
            json_path=json_path,
            prefix="",
        )

        active = logger.active_backends
        print(f"[logging] active backends: {', '.join(active) if active else 'none'}", flush=True)
        return logger

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Send metrics to every active backend."""
        prefixed = _apply_prefix(metrics, self._prefix)

        if self._use_stdout:
            _print_step(step, prefixed)

        if self._experiment is not None:
            self._experiment.log_metrics(prefixed, step=step)

        if self._json_handle is not None:
            record = {"step": step, **prefixed}
            self._json_handle.write(json.dumps(record) + "\n")

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self._experiment is not None:
            self._experiment.log_parameters(params)
        if self._json_handle is not None:
            self._json_handle.write(json.dumps({"_hyperparams": params}) + "\n")

    def log_text(self, key: str, text: str, step: int) -> None:
        if self._experiment is not None:
            self._experiment.log_text(text, step=step, metadata={"key": key})
        if self._json_handle is not None:
            self._json_handle.write(json.dumps({"step": step, "_text": {key: text}}) + "\n")

    def finish(self) -> None:
        if self._experiment is not None:
            self._experiment.end()
        if self._json_handle is not None:
            self._json_handle.close()
            self._json_handle = None

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finish()


    @property
    def active_backends(self) -> list[str]:
        backends = []
        if self._use_stdout:
            backends.append("stdout")
        if self._experiment is not None:
            backends.append("comet")
        if self._json_path is not None:
            backends.append(f"json({self._json_path})")
        return backends


def _init_comet(config: dict):
    comet_cfg = config.get("comet", {})
    import warnings as _warnings
    try:
        import comet_ml

        experiment = comet_ml.Experiment(
            api_key=comet_cfg.get("api_key"),
            project_name=comet_cfg.get("project_name", "fp8_tis_gsm8k"),
            workspace=comet_cfg.get("workspace"),
        )
        if comet_cfg.get("experiment_name"):
            experiment.set_name(comet_cfg["experiment_name"])
        for tag in comet_cfg.get("tags", []):
            experiment.add_tag(tag)

        experiment.log_parameters(_flatten_dict(config))
        return experiment

    except ImportError:
        _warnings.warn(
            "comet_ml is not installed — Comet backend disabled.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    except ValueError as e:
        _warnings.warn(
            f"Comet-ML init failed (missing API key?) — Comet backend disabled.\n"
            f"Set COMET_API_KEY env var or add api_key: under the comet: config block.\n"
            f"Original error: {e}",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    except Exception as e:
        _warnings.warn(
            f"Comet-ML init failed — Comet backend disabled. Error: {e}",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _apply_prefix(metrics: dict, prefix: str) -> dict:
    if not prefix:
        return metrics
    return {
        (prefix + k if not k.startswith(prefix) else k): v
        for k, v in metrics.items()
    }


def _print_step(step: int, metrics: dict) -> None:
    row = "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(metrics.items())
    )
    print(f"[step {step:>6}]  {row}", flush=True)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def gsm8k_metrics(
    predictions: list[str],
    gold_answers: list[str],
    rewards: list[float],
    group_size: int = 1,
) -> dict:
    from data.gsm8k import extract_solution

    n = len(predictions)

    # answer quality
    n_parsed = sum(1 for p in predictions if extract_solution(p, "strict") is not None)
    n_marker = sum(1 for p in predictions if "####" in p)

    # reward statistics (mirrors QeRL reward / reward_std)
    mean_reward = sum(rewards) / max(n, 1)
    if n > 1:
        import math
        variance = sum((r - mean_reward) ** 2 for r in rewards) / (n - 1)
        std_reward = math.sqrt(variance)
    else:
        std_reward = 0.0

    # frac_reward_zero_std: fraction of groups where all responses got the same
    # reward (advantage = 0, gradient = 0). High value → model is stuck.
    frac_zero_std = 0.0
    if group_size > 1 and n % group_size == 0:
        n_groups = n // group_size
        zero_std_groups = 0
        for i in range(n_groups):
            group_rewards = rewards[i * group_size : (i + 1) * group_size]
            if max(group_rewards) == min(group_rewards):
                zero_std_groups += 1
        frac_zero_std = zero_std_groups / n_groups

    return {
        "reward/mean":               mean_reward,
        "reward/std":                std_reward,
        "reward/frac_zero_std":      frac_zero_std,   # QeRL: frac_reward_zero_std
        "answer/parse_success_rate": n_parsed / max(n, 1),
        "answer/final_marker_rate":  n_marker / max(n, 1),
    }


def response_length_metrics(response_mask: "torch.Tensor") -> dict:  # noqa: F821
    import torch
    lengths = response_mask.sum(dim=-1).float()
    return {
        "response_length/mean": lengths.mean().item(),
        "response_length/max":  lengths.max().item(),
        "response_length/min":  lengths.min().item(),
    }


def memory_metrics() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu/memory_allocated":     torch.cuda.memory_allocated() / 1e9,
                "gpu/memory_reserved":      torch.cuda.memory_reserved() / 1e9,
                "gpu/max_memory_allocated": torch.cuda.max_memory_allocated() / 1e9,
            }
    except Exception:
        pass
    return {}


def throughput_metrics(n_tokens: int, elapsed_seconds: float, prefix: str = "rollout") -> dict:
    tps = n_tokens / max(elapsed_seconds, 1e-6)
    return {f"{prefix}/tokens_per_second": tps}


class Timer:
    def __init__(self):
        self.elapsed = 0.0
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        assert self._start is not None
        self.elapsed = time.perf_counter() - self._start
