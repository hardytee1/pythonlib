from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional


__all__ = [
    "CommandExecutionError",
    "StartOptions",
    "start_cluster",
    "show_status",
    "deploy_model",
]


class CommandExecutionError(RuntimeError):
    """Raised when an underlying Ray command fails."""


@dataclass
class StartOptions:
    head: bool
    worker: bool
    address: Optional[str]
    dashboard_host: str = "0.0.0.0"


def _run_ray_subprocess(*args: str) -> None:
    """Invoke the Ray CLI and stream output."""
    ray_executable = shutil.which("ray")
    command = [ray_executable, *args] if ray_executable else [sys.executable, "-m", "ray", *args]

    try:
        subprocess.run(command, check=True)
        return
    except FileNotFoundError:
        fallback = [sys.executable, "-m", "ray", *args]
        try:
            subprocess.run(fallback, check=True)
            return
        except subprocess.CalledProcessError as exc:  # pragma: no cover - bubble up context for CLI
            raise CommandExecutionError(f"Ray command failed: {' '.join(fallback)}") from exc
    except subprocess.CalledProcessError as exc:
        raise CommandExecutionError(f"Ray command failed: {' '.join(command)}") from exc


def start_cluster(options: StartOptions) -> None:
    """Start a Ray head or worker node based on the provided options."""
    if options.head == options.worker:
        raise ValueError("Specify exactly one of --head or --worker")

    if options.head:
        _run_ray_subprocess("start", "--head", f"--dashboard-host={options.dashboard_host}")
        return

    if not options.address:
        raise ValueError("--address is required when starting a worker node")

    sanitized = options.address.strip("'\"")
    _run_ray_subprocess("start", f"--address={sanitized}")


def show_status() -> None:
    """Display the current Ray cluster status."""
    _run_ray_subprocess("status")


def deploy_model(
    model_source: str,
    model_id: Optional[str] = None,
    http_host: str = "0.0.0.0",
    http_port: int = 8000,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 22000,
    runtime_interface: Optional[str] = None,
) -> None:
    """Deploy an LLM using Ray Serve with sensible defaults."""

    # try:
        
    # except ImportError as exc:  # pragma: no cover - dependency missing
    #     raise RuntimeError("Ray Serve is required. Install the 'ray[serve]' extra.") from exc

    from ray import serve
    from ray.serve.llm import LLMConfig, build_openai_app
    inferred_id = model_id or _derive_model_id(model_source)
    env_vars: Dict[str, str] = {"VLLM_USE_V1": "1"}

    env_vars["GLOO_SOCKET_IFNAME"] = "enp132s0"
    env_vars["NCCL_SOCKET_IFNAME"] = "enp132s0"
    # if runtime_interface:
    #     env_vars["GLOO_SOCKET_IFNAME"] = runtime_interface
    #     env_vars["NCCL_SOCKET_IFNAME"] = runtime_interface

    serve.start(http_options={"host": http_host, "port": http_port})

    llm_config = LLMConfig(
        model_loading_config={
            "model_id": inferred_id,
            "model_source": model_source,
        },
        deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
        engine_kwargs={
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        },
        runtime_env={"env_vars": env_vars},
    )

    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app, blocking=True)


def _derive_model_id(model_source: str) -> str:
    """Produce an identifier when the caller omits --model-id."""
    candidate = model_source.rsplit("/", maxsplit=1)[-1]
    sanitized = candidate.replace("-Instruct", "").replace("_", "-")
    return sanitized or "llm"
