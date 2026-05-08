"""
后续实验调度: Vanilla-PPO baseline 验证 + K=5 每 slice tau_k.

实验:
  - Vanilla-PPO baseline 验证 (检查 reference U 能否复现).
  - K=5 用 per-slice tau_k (URLLC 收紧到 tau=0.3).

预估耗时:
  串行:     ~10-15 min
  2 worker: ~6-10 min

Usage:
    python run_followup_experiments.py --workers 2

输出结构:
    experiments/<run_name>/
        experiment_config.json
        summary.md
        vanilla_verify/
        k5_per_slice_tau/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


EXPERIMENTS = [
    ("vanilla_verify", "Vanilla-PPO baseline verification",
     "core/run_vanilla_verify.py", "2-5 min"),
    ("k5_per_slice_tau", "K=5 with per-slice tau_k (tighter URLLC SLA)",
     "core/run_k5_per_slice_tau.py", "4-8 min"),
]


def run_experiment(name, label, script_path, output_dir, workers, log_file):
    print(f"\n{'='*70}")
    print(f"  [{name}] {label}")
    print(f"  Script: {script_path}")
    print(f"  Workers: {workers}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    sub_output = output_dir / name
    sub_output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        with open(log_file, 'w') as f:
            subprocess.run([
                sys.executable, script_path,
                "--output", str(sub_output),
                "--workers", str(workers),
            ], stdout=f, stderr=subprocess.STDOUT, check=True)
        elapsed = time.time() - t0
        print(f"  [{name}] OK ({elapsed/60:.1f} min)")
        return {"name": name, "status": "OK", "elapsed_sec": elapsed}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"  [{name}] FAILED ({elapsed/60:.1f} min); see {log_file}")
        return {"name": name, "status": "FAILED",
                "elapsed_sec": elapsed, "returncode": e.returncode}


def write_summary(output_dir, summary, config):
    md_lines = [
        "# Follow-up experiment suite: master summary",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Workers: {config['workers']}",
        "",
        "## Per-experiment status",
        "",
        "| Experiment | Status | Time (min) | Output |",
        "|---|---|---|---|",
    ]
    for s in summary:
        out_path = output_dir / s["name"]
        md_lines.append(
            f"| {s['name']} | {s['status']} | "
            f"{s['elapsed_sec']/60:.1f} | "
            f"`{out_path}/summary_stats.md` |"
        )

    md_lines.extend([
        "",
        "## Aggregate findings",
        "",
    ])

    for s in summary:
        if s["status"] != "OK":
            continue
        sub_path = output_dir / s["name"] / "summary_stats.json"
        if sub_path.exists():
            with open(sub_path) as f:
                sub = json.load(f)
            md_lines.append(f"### {s['name']}")
            md_lines.append("")
            md_lines.append(f"```json\n{json.dumps(sub, indent=2)}\n```")
            md_lines.append("")

    md_lines.append("")

    with open(output_dir / "summary.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Follow-up supplementary experiments"
    )
    parser.add_argument("--workers", type=int, default=2,
                        help="Workers per experiment (default 2)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output or f"experiments/followup_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_to_run = EXPERIMENTS
    if args.only:
        experiments_to_run = [e for e in EXPERIMENTS if e[0] == args.only]
        if not experiments_to_run:
            print(f"ERROR: --only '{args.only}' not in experiment list.")
            sys.exit(1)
    elif args.skip:
        experiments_to_run = [e for e in EXPERIMENTS if e[0] not in args.skip]

    config = {
        "timestamp": timestamp,
        "workers": args.workers,
        "experiments": [e[0] for e in experiments_to_run],
        "skipped": args.skip,
        "purpose": "Follow-up supplementary experiments",
    }
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# Follow-up experiment suite: master run")
    print(f"# Output: {output_dir}")
    print(f"# Experiments: {[e[0] for e in experiments_to_run]}")
    print(f"# Workers: {args.workers}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    summary = []
    t_total = time.time()
    for name, label, script, est in experiments_to_run:
        log_file = output_dir / f"{name}.log"
        result = run_experiment(name, label, script, output_dir, args.workers, log_file)
        summary.append(result)

    elapsed_total = time.time() - t_total
    print(f"\n{'#'*70}")
    print(f"# Follow-up experiment suite: complete")
    print(f"# Total wall: {elapsed_total/60:.1f} min")
    print(f"# Status: {sum(1 for s in summary if s['status'] == 'OK')}/{len(summary)} OK")
    print(f"# See {output_dir}/summary.md")
    print(f"{'#'*70}\n")

    write_summary(output_dir, summary, config)
    with open(output_dir / "run_summary.json", 'w') as f:
        json.dump({"config": config, "results": summary,
                   "elapsed_total_sec": elapsed_total}, f, indent=2)


if __name__ == "__main__":
    main()
