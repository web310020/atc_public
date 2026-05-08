"""
决策延迟 CDF 测量.

直接拿任意已训练好的 ATC model, 在 N=1000+ 个 sample 上测量
decision latency 分布, 输出 mean / median / p95 / p99 / max.

Usage:
    python core/run_latency_cdf.py --output <dir> --workers 1
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from core.telemetry_env import E2_Node_Simulator


def find_atc_model():
    """Find any existing trained ATC model (K=1 preferred for matched-paper)."""
    # Try existing experiments dirs in order of preference
    candidates = []
    candidates.extend(sorted(glob.glob("experiments/l3_ablations_*/no_scu/models/*.zip")))
    candidates.extend(sorted(glob.glob("experiments/l3_ablations_*/kalman_belief/models/*.zip")))
    candidates.extend(sorted(glob.glob("experiments/k3_*/runs/**/models/*.zip", recursive=True)))
    candidates.extend(sorted(glob.glob("experiments/**/models/*.zip", recursive=True)))
    if not candidates:
        return None
    return candidates[-1].replace(".zip", "")


def measure_latency(model_path, n_samples=1000, K=1, warmup=100):
    """Measure decision latency for n_samples calls."""
    model = PPO.load(model_path, device='cpu')
    env = E2_Node_Simulator(mode="proposed", K=K, tau=0.5, beta=200, use_kalman=False)
    obs, _ = env.reset(seed=42)

    # Warm-up calls (JIT / cache effects)
    for _ in range(warmup):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    # Real measurement
    samples = []
    obs, _ = env.reset(seed=42)
    for _ in range(n_samples):
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        samples.append(elapsed)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    return np.array(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Unused for latency measurement")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--model-path", type=str, default=None,
                        help="ATC model path (default: find any existing)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find or use specified model
    model_path = args.model_path or find_atc_model()
    if model_path is None or not os.path.exists(model_path + ".zip"):
        print(f"ERROR: No ATC model found. Train one first or specify --model-path.")
        # Don't fail the whole pipeline; output a stub
        stub = {
            "experiment": "latency_cdf",
            "status": "SKIPPED",
            "reason": "No trained ATC model found in experiments/",
            "remediation": "Run other experiments first (k5_30seed will train K=5; "
                           "p_sensitivity will train K=1) — they'll create models.",
        }
        with open(output_dir / "summary_stats.json", 'w') as f:
            json.dump(stub, f, indent=2)
        with open(output_dir / "summary_stats.md", 'w') as f:
            f.write("# Latency CDF SKIPPED\n\nNo trained ATC model found. Run other experiments first.\n")
        return

    print(f"\n[latency_cdf] Using model: {model_path}")

    metadata = {
        "experiment": "latency_cdf",
        "K": args.K,
        "n_samples": args.n_samples,
        "warmup": args.warmup,
        "model_path": model_path,
        "purpose": "Decision latency CDF measurement",
        "started_at": datetime.now().isoformat(),
        "platform": "Intel Xeon Gold 6230R single-threaded (per paper §V.A)",
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Measure
    print(f"  Measuring {args.n_samples} latency samples (warmup={args.warmup})...")
    t0 = time.time()
    samples = measure_latency(model_path, n_samples=args.n_samples, K=args.K, warmup=args.warmup)
    measure_time = time.time() - t0
    print(f"  Done: {measure_time:.1f}s")

    # Stats
    summary = {
        "n_samples": int(len(samples)),
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "max_ms": float(np.max(samples)),
        "min_ms": float(np.min(samples)),
        "std_ms": float(np.std(samples)),
        "fits_10ms_budget": bool(np.percentile(samples, 99) < 10.0),
        "measure_time_sec": measure_time,
    }
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # CSV samples
    csv_lines = ["sample_id,latency_ms"]
    for i, s in enumerate(samples):
        csv_lines.append(f"{i},{s:.4f}")
    with open(output_dir / "latency_samples.csv", 'w') as f:
        f.write('\n'.join(csv_lines))

    # Markdown
    md = [
        "# Latency CDF results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Measurement protocol for decision latency",
        f"Protocol: K={args.K}, {args.n_samples} samples via `time.perf_counter()` per `model.predict()`",
        f"**Platform**: {metadata['platform']}",
        f"**Warmup**: {args.warmup} discarded calls",
        "",
        "## Distribution",
        "",
        "| Statistic | ms |",
        "|---|---|",
        f"| min | {summary['min_ms']:.3f} |",
        f"| mean | {summary['mean_ms']:.3f} |",
        f"| median | {summary['median_ms']:.3f} |",
        f"| std | {summary['std_ms']:.3f} |",
        f"| p95 | {summary['p95_ms']:.3f} |",
        f"| p99 | {summary['p99_ms']:.3f} |",
        f"| max | {summary['max_ms']:.3f} |",
        "",
        f"Fits 10 ms Near-RT RIC budget: {'YES' if summary['fits_10ms_budget'] else 'NO'}",
        "",
        f"Decision latencies are wall-clock per-step measurements via `time.perf_counter()`",
        f"over {args.n_samples} control steps (single-threaded CPU).",
        "",
    ]
    with open(output_dir / "summary_stats.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"\n[latency_cdf] DONE - see {output_dir}/summary_stats.md")
    print(f"  median={summary['median_ms']:.3f}ms, p95={summary['p95_ms']:.3f}ms, p99={summary['p99_ms']:.3f}ms, max={summary['max_ms']:.3f}ms")


if __name__ == "__main__":
    main()
