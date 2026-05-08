"""
补充实验 suite 主调度脚本.

把 4 个实验跑到同一个 timestamp 文件夹里:
  A. K=5 30-seed 扩展
  B. P-kernel 校准敏感性
  C. beta 敏感性扫描
  D. Latency CDF 测量

预估耗时:
  串行:        ~12-18 h
  4 worker:    ~4-8 h
  6 worker:    ~3-6 h

Usage:
    python run_supplementary_experiments.py --workers 4
    python run_supplementary_experiments.py --workers 6 --skip latency_cdf
    python run_supplementary_experiments.py --output experiments/my_run

输出结构:
    experiments/<run_name>/
        experiment_config.json
        summary.md
        k5_30seed/
        p_sensitivity/
        beta_sensitivity/
        latency_cdf/
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
    ("k5_30seed", "K=5 30-seed extension",
     "core/run_k5_30seed.py", "4-6 hr"),
    ("p_sensitivity", "P-kernel calibration sensitivity sweep",
     "core/run_p_sensitivity.py", "1-2 hr"),
    ("beta_sensitivity", "beta sensitivity sweep",
     "core/run_beta_sensitivity.py", "5-7 hr"),
    ("latency_cdf", "Latency CDF measurement",
     "core/run_latency_cdf.py", "0.5 hr"),
]


def run_experiment(name, label, script_path, output_dir, workers, log_file):
    """Run one experiment subprocess; capture output + status."""
    print(f"\n{'='*70}")
    print(f"  [{name}] {label}")
    print(f"  Script: {script_path}")
    print(f"  Workers: {workers}")
    print(f"  Output: {output_dir}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    sub_output = output_dir / name
    sub_output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run([
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
    """Aggregate per-experiment summaries into master markdown."""
    md_lines = [
        "# Experiment suite: master summary",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total experiments: {len(summary)}",
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
        "## Aggregate findings (read each summary_stats.md for detail)",
        "",
    ])

    for s in summary:
        if s["status"] != "OK":
            continue
        sub_summary_path = output_dir / s["name"] / "summary_stats.json"
        if sub_summary_path.exists():
            with open(sub_summary_path) as f:
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
        description="Run all supplementary experiments to a single output folder"
    )
    parser.add_argument("--workers", type=int, default=2,
                        help="Workers per experiment (default 2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default experiments/supplementary_<timestamp>)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip these experiments (e.g. --skip latency_cdf)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this experiment (e.g. --only k5_30seed)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output or f"experiments/supplementary_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter experiments
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
        "purpose": "Sensitivity and supplementary experiment suite",
        "estimated_total_hr": sum_hours([e[3] for e in experiments_to_run]),
    }
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# Experiment suite: master run")
    print(f"# Output: {output_dir}")
    print(f"# Experiments: {[e[0] for e in experiments_to_run]}")
    print(f"# Workers: {args.workers}")
    print(f"# Estimated total: {config['estimated_total_hr']}")
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
    print(f"# Experiment suite: complete")
    print(f"# Total wall time: {elapsed_total/3600:.2f} hr ({elapsed_total/60:.1f} min)")
    print(f"# Status: {sum(1 for s in summary if s['status'] == 'OK')}/{len(summary)} OK")
    print(f"# See {output_dir}/summary.md")
    print(f"{'#'*70}\n")

    write_summary(output_dir, summary, config)

    with open(output_dir / "run_summary.json", 'w') as f:
        json.dump({"config": config, "results": summary,
                   "elapsed_total_sec": elapsed_total}, f, indent=2)


def sum_hours(est_strs):
    """Parse '4-6 hr' strings into total range estimate."""
    lo, hi = 0, 0
    for s in est_strs:
        s = s.replace(" hr", "").strip()
        if "-" in s:
            a, b = s.split("-")
            lo += float(a); hi += float(b)
        else:
            lo += float(s); hi += float(s)
    return f"{lo:.1f}-{hi:.1f} hr (serial; parallel may be 2-3x faster)"


if __name__ == "__main__":
    main()
