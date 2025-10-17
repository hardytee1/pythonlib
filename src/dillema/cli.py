import argparse
import sys
from typing import Optional, Sequence

from .commands import (
    CommandExecutionError,
    StartOptions,
    deploy_model,
    show_status,
    start_cluster,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="dillema", description="Convenient wrappers around the Ray CLI and Serve APIs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start a Ray head or worker node or run the web app")
    role_group = start_parser.add_mutually_exclusive_group(required=False)
    role_group.add_argument("--head", action="store_true", help="Start the Ray head node with dashboard enabled")
    role_group.add_argument("--worker", action="store_true", help="Start a Ray worker node that joins an existing cluster")
    start_parser.add_argument("--web", action="store_true", help="Run the bundled FastAPI web app (equivalent to uvicorn app.main:app --host/--port)")
    start_parser.add_argument("--address", help="Ray head address the worker should join (required for --worker)")
    start_parser.add_argument(
        "--dashboard-host",
        default="0.0.0.0",
        help="Host interface for the Ray dashboard when starting the head node",
    )
    start_parser.add_argument("--web-host", default="0.0.0.0", help="Host for the web server if --web is used")
    start_parser.add_argument("--web-port", type=int, default=8000, help="Port for the web server if --web is used")
    start_parser.add_argument("--no-docker", action="store_true", help="Skip running `docker compose up -d` before starting web")
    start_parser.add_argument("--no-migrate", action="store_true", help="Skip running `alembic upgrade head` before starting web")
    start_parser.add_argument("--no-npm-build", action="store_true", help="Skip running `npm run build` after starting web")

    subparsers.add_parser("status", help="Show the current Ray cluster status")

    deploy_parser = subparsers.add_parser("deploy", help="Launch an LLM deployment with Ray Serve")
    deploy_parser.add_argument("--model", required=True, help="Model source identifier, e.g. Qwen/Qwen2.5-14B-Instruct-AWQ")
    deploy_parser.add_argument("--model-id", help="Override the model identifier reported to Ray Serve")
    deploy_parser.add_argument("--http-host", default="0.0.0.0", help="HTTP host for Ray Serve")
    deploy_parser.add_argument("--http-port", type=int, default=8000, help="HTTP port for Ray Serve")
    deploy_parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallelism for the LLM engine")
    deploy_parser.add_argument("--pipeline-parallel", type=int, default=2, help="Pipeline parallelism for the LLM engine")
    deploy_parser.add_argument("--gpu-mem", type=float, default=0.9, help="GPU memory utilization ratio (0-1)")
    deploy_parser.add_argument("--max-model-len", type=int, default=22000, help="Maximum model sequence length")
    deploy_parser.add_argument(
        "--runtime-interface",
        help="Network interface to expose for collective backends (sets NCCL/GLOO socket IFNAME)",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "start":
            # If --web was requested, run the FastAPI app instead of starting Ray
            if getattr(args, "web", False):
                # Run optional preparatory steps, then launch the FastAPI app via uvicorn,
                # and finally run the frontend build.
                import subprocess
                import shlex

                # determine working directory for web app commands. Prefer apps/RAGforge if present
                from pathlib import Path

                repo_root = Path.cwd()
                ragforge_dir = repo_root / "apps" / "RAGforge"
                web_cwd = ragforge_dir if ragforge_dir.exists() and ragforge_dir.is_dir() else repo_root

                # 1) docker compose up -d (run in web_cwd so Dockerfile/docker-compose in apps/RAGforge is used)
                if not getattr(args, "no_docker", False):
                    try:
                        print(f"Running: docker compose up -d (cwd={web_cwd})")
                        subprocess.run(shlex.split("sudo docker compose up -d"), check=True, cwd=str(web_cwd))
                    except subprocess.CalledProcessError as exc:
                        print(f"docker compose failed: {exc}")
                        raise

                # 2) alembic upgrade head (run in the same cwd so migrations config is found)
                if not getattr(args, "no_migrate", False):
                    try:
                        print(f"Running: alembic upgrade head (cwd={web_cwd})")
                        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True, cwd=str(web_cwd))
                    except subprocess.CalledProcessError as exc:
                        print(f"alembic upgrade failed: {exc}")
                        raise

                # 3) start uvicorn app.main:app
                web_cmd = [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    args.web_host,
                    "--port",
                    str(args.web_port),
                ]
                uvicorn_proc = subprocess.Popen(web_cmd)

                # 4) npm run build (fire-and-forget; wait for it to complete)
                if not getattr(args, "no_npm_build", False):
                    try:
                        print(f"Running: npm run build (cwd={web_cwd})")
                        subprocess.run(shlex.split("npm run build"), check=True, cwd=str(web_cwd))
                    except subprocess.CalledProcessError as exc:
                        print(f"npm run build failed: {exc}")
                        # stop uvicorn if build fails
                        uvicorn_proc.terminate()
                        raise

                # wait for uvicorn to exit (blocks until server stops)
                uvicorn_proc.wait()
                return

            start_cluster(
                StartOptions(
                    head=args.head,
                    worker=args.worker,
                    address=args.address,
                    dashboard_host=args.dashboard_host,
                )
            )
        elif args.command == "status":
            show_status()
        elif args.command == "deploy":
            deploy_model(
                model_source=args.model,
                model_id=args.model_id,
                http_host=args.http_host,
                http_port=args.http_port,
                tensor_parallel_size=args.tensor_parallel,
                pipeline_parallel_size=args.pipeline_parallel,
                gpu_memory_utilization=args.gpu_mem,
                max_model_len=args.max_model_len,
                runtime_interface=args.runtime_interface,
            )
        else:  # pragma: no cover - argparse ensures command is valid
            parser.error("Unknown command")
    except (ValueError, CommandExecutionError, RuntimeError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
