from __future__ import annotations

import builtins
import hashlib
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from triton.runtime.cache import get_cache_manager, triton_key


_COSTMODEL_MEM_CACHE: Dict[str, float] = {}


def candidate_tritonsim_opts() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("TRITONSIM_OPT")
    if env_path:
        candidates.append(Path(env_path))

    repo_root = Path(__file__).resolve().parents[4]
    candidates.extend(
        [
            repo_root / "third_party" / "vTriton" / "build" / "bin" / "tritonsim-opt",
            repo_root / "third_party" / "vTriton" / "build" / "tritonsim" / "bin" / "tritonsim-opt",
        ]
    )

    machine = platform.machine().lower()
    preferred_build_dir = "build_arm64" if machine in {"aarch64", "arm64"} else "build_x86"
    candidates.append(repo_root / "third_party" / "vTriton" / preferred_build_dir / "tritonsim" / "bin" / "tritonsim-opt")
    candidates.extend(
        [
            repo_root / "third_party" / "vTriton" / "build_arm64" / "tritonsim" / "bin" / "tritonsim-opt",
            repo_root / "third_party" / "vTriton" / "build_x86" / "tritonsim" / "bin" / "tritonsim-opt",
            repo_root / "third_party" / "vTriton" / "build_b5cc" / "bin" / "tritonsim-opt",
        ]
    )
    return candidates


def resolve_tritonsim_opt() -> str:
    for candidate in candidate_tritonsim_opts():
        if candidate and candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find tritonsim-opt. Set TRITONSIM_OPT env var or build vTriton."
    )


def capture_ttir(launch):
    from triton.backends.ascend import compiler as ascend_compiler

    ascend_compiler._costmodel_ttir = None
    launch()
    ttir = ascend_compiler._costmodel_ttir
    if ttir is None:
        raise RuntimeError(
            "Costmodel backend was enabled but compiler did not capture TTIR. "
            "Ensure enable_costmodel_backend=True and the kernel was compiled."
        )
    return ttir


def run_costmodel(ttir_or_path, extra_args=None, dump_ir_on_error=False):
    cmd = [resolve_tritonsim_opt()]
    if extra_args:
        cmd.extend(extra_args)
    if "-allow-unregistered-dialect" not in cmd:
        cmd.append("-allow-unregistered-dialect")

    if os.path.exists(str(ttir_or_path)):
        cmd.append(str(ttir_or_path))
        stdin_input = None
    else:
        cmd.append("-")
        stdin_input = ttir_or_path

    try:
        result = subprocess.run(cmd, input=stdin_input, capture_output=True, text=True, check=True)
        if result.stderr:
            print(result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败，返回码: {exc.returncode}")
        print(f"错误输出: {exc.stderr}")
        if dump_ir_on_error and not stdin_input:
            print(f"IR 文件: {ttir_or_path}")
        return None
    except FileNotFoundError:
        print("找不到该二进制文件或命令")
        return None


def get_costmodel_jobs(num_tasks: int) -> int:
    if num_tasks <= 1:
        return 1
    raw = os.environ.get("TRITON_COSTMODEL_JOBS")
    if raw is not None:
        try:
            parsed = int(raw)
            if parsed > 0:
                return min(parsed, num_tasks)
        except Exception:
            pass
    default_jobs = os.cpu_count() or 1
    return min(max(1, default_jobs), num_tasks)


def make_costmodel_cache_key(ttir: str, extra_args: Optional[List[str]]) -> str:
    h = hashlib.sha256()
    h.update(ttir.encode("utf-8"))
    h.update(b"|")
    if extra_args:
        h.update(" ".join(extra_args).encode("utf-8"))
    h.update(b"|")
    try:
        h.update(resolve_tritonsim_opt().encode("utf-8"))
    except Exception:
        h.update(b"tritonsim-unknown")
    return h.hexdigest()


def load_costmodel_latency(cache_key: str) -> Optional[float]:
    cached = _COSTMODEL_MEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_manager = get_cache_manager(f"costmodel_{triton_key()}")
    file_name = f"{cache_key}.json"
    payload = cache_manager.get_file(file_name)
    if payload is None:
        return None

    try:
        parsed = json.loads(payload)
        latency = float(parsed["latency"])
        _COSTMODEL_MEM_CACHE[cache_key] = latency
        return latency
    except Exception:
        return None


def store_costmodel_latency(cache_key: str, latency: float) -> None:
    _COSTMODEL_MEM_CACHE[cache_key] = latency
    cache_manager = get_cache_manager(f"costmodel_{triton_key()}")
    file_name = f"{cache_key}.json"
    cache_manager.put(json.dumps({"latency": latency}), file_name, binary=False)


def parse_latency(stdout: str) -> float:
    import re

    match = re.search(r"Estimated Time:\s+([0-9.]+)\s*us", stdout)
    return float(match.group(1)) if match else float("inf")


def costmodel_bench(autotuner, *args, pruned_configs, key, **kwargs):
    """Evaluate all candidate configs via costmodel and expose config->time map."""
    costmodel_latencies = {}
    bench_start = time.time()

    from triton.backends.ascend import compiler as ascend_compiler

    pending_items = []
    for config in pruned_configs:
        ascend_compiler._costmodel_ttir = None
        current = dict(kwargs, **config.all_kwargs())
        current["warmup"] = True
        try:
            autotuner.fn.run(*args, **current)
        except Exception:
            costmodel_latencies[config] = float("inf")
            continue

        ttir = ascend_compiler._costmodel_ttir
        if ttir is None:
            costmodel_latencies[config] = float("inf")
            continue
        pending_items.append((config, ttir))

    ascend_compiler._costmodel_ttir = None

    def eval_one(item):
        config, ttir = item
        extra_args = ["-ascend-perf-model"]
        cache_key = make_costmodel_cache_key(ttir, extra_args)
        cached = load_costmodel_latency(cache_key)
        if cached is not None:
            return config, cached

        output = run_costmodel(ttir_or_path=ttir, extra_args=extra_args)
        latency = float("inf") if output is None else parse_latency(output)
        store_costmodel_latency(cache_key, latency)
        return config, latency

    if pending_items:
        jobs = get_costmodel_jobs(len(pending_items))
        if jobs <= 1:
            for item in pending_items:
                cfg, latency = eval_one(item)
                costmodel_latencies[cfg] = latency
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(eval_one, item) for item in pending_items]
                for future in as_completed(futures):
                    try:
                        cfg, latency = future.result()
                        costmodel_latencies[cfg] = latency
                    except Exception:
                        pass

    for cfg in pruned_configs:
        costmodel_latencies.setdefault(cfg, float("inf"))

    autotuner.bench_time = time.time() - bench_start
    autotuner.configs_timings = costmodel_latencies

    valid = [cfg for cfg, t in costmodel_latencies.items() if t != float("inf")]
    if valid:
        autotuner.cache[key] = builtins.min(valid, key=lambda c: costmodel_latencies[c])
    else:
        autotuner.cache[key] = pruned_configs[0]

    return costmodel_latencies
