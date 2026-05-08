#!/usr/bin/env python
# run_k3_experiment.py
# ============================================================
# K=3 多 slice 训练 + eval runner.
#
# 流程:
#   1. 训练 ATC ("proposed") mode: Dirichlet PPO + per-slice Lagrangian,
#      K=3 配置.
#   2. 训练 Vanilla-PPO baseline: 同样的 Dirichlet policy, 但去掉
#      belief engine / trust fusion / adaptive telemetry. RL stack 一致,
#      所以 ATC vs baseline 是受控对比.
#   3. 在确定性 eval set 上评估两个 mode, 输出 per-slice violation rate,
#      aggregate U, 以及一个决策块 (PROMOTE / FALLBACK / AMBIGUOUS),
#      判定标准在下面定义.
#
# Output: JSON + markdown report 写到 experiments/<run_name>/
# ============================================================

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch

# Ensure relative imports work when running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.k3_env import make_env, TEMPLATES
from core.k3_dirichlet_ppo import K3DirichletPPO


# ============================================================
# Decision criteria — copied verbatim from experiment design brief
# ============================================================

SUCCESS_CRITERIA = {
    # Most-constrained slice violation rate: ATC at least 2x lower than Vanilla
    "most_constrained_ratio": 2.0,
    # Aggregate U: ATC within 10% of Vanilla
    "agg_u_tolerance": 0.10,
    # Training stability: all seeds must produce finite metrics
    "min_stable_seeds_fraction": 1.0,  # 3/3 seeds must converge
}

AMBIGUOUS_CRITERIA = {
    "min_stable_seeds_fraction": 2.0 / 3.0,   # 2/3 seeds ok -> AMBIGUOUS
    "most_constrained_ratio_min": 1.3,        # >=1.3x but <2x -> AMBIGUOUS
}


# ============================================================
# Training & evaluation for a single (mode, seed)
# ============================================================

def run_one(mode: str, template: str, seed: int, total_steps: int,
            use_kalman: bool, out_dir: Path,
            use_lagrangian: bool = True,
            normalize_per_slice_rewards: bool = True,
            lr_dual: float = 1e-3,
            alpha_floor: float = 1.0,
            sla_budget_override: list = None) -> dict:
    env_fn = make_env(mode=mode, template=template, use_kalman=use_kalman)
    sla_budget = (sla_budget_override if sla_budget_override is not None
                  else TEMPLATES[template]["sla_budget"])

    tag = f"{mode}_seed{seed}"
    per_run_dir = out_dir / "runs" / tag
    per_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> TRAIN  {tag}  template={template} steps={total_steps} "
          f"lagr={use_lagrangian} rnorm={normalize_per_slice_rewards} "
          f"lr_dual={lr_dual} alpha_floor={alpha_floor} sla_budget={sla_budget}")
    trainer = K3DirichletPPO(
        env_fn=env_fn,
        K=len(TEMPLATES[template]["names"]),  # K-agnostic: derived from template
        sla_budget=sla_budget,
        total_steps=total_steps,
        seed=seed,
        use_lagrangian=use_lagrangian,
        normalize_per_slice_rewards=normalize_per_slice_rewards,
        lr_dual=lr_dual,
        alpha_floor=alpha_floor,
        verbose=True,
        log_dir=str(per_run_dir),          # streams training_log.jsonl + periodic ckpts
        checkpoint_every_updates=10,
        tag=tag,
    )
    t_start = time.time()
    try:
        train_log = trainer.train()
        status = "ok"
        error = None
    except Exception as e:
        train_log = trainer.train_log
        status = "error"
        error = f"{type(e).__name__}: {e}"
        print(f"!!! TRAINING ERROR ({tag}): {error}")
        traceback.print_exc()
        # Persist what we have so the crash is debuggable later
        try:
            (per_run_dir / "CRASH.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
    train_time = time.time() - t_start

    # Dump the full in-memory train_log as a single JSON (complement to JSONL stream)
    try:
        (per_run_dir / f"{tag}_training_log.json").write_text(
            json.dumps(train_log, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"  (train_log JSON save failed: {e})")

    # Eval even on error (may still have partial policy)
    try:
        print(f">>> EVAL   {tag}  deterministic")
        eval_metrics = trainer.evaluate(
            n_episodes=10, deterministic=True,
            trajectory_csv_path=str(per_run_dir / f"{tag}_eval_trajectory.csv"),
        )
        # Also run stochastic eval (5 episodes) so we can see action spread
        stoch_metrics = trainer.evaluate(
            n_episodes=5, deterministic=False,
            trajectory_csv_path=str(per_run_dir / f"{tag}_eval_stochastic_trajectory.csv"),
        )
        eval_metrics["stochastic"] = stoch_metrics
    except Exception as e:
        eval_metrics = {"error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()
        status = "eval_error" if status == "ok" else status

    # Seed-level stability check:
    # mark as "diverged" if any per-slice violation rate == 1.0
    # (policy stuck in a maximally-bad regime) OR returns all -inf
    stable = (status == "ok" and
              "per_slice_viol_rate" in eval_metrics and
              all(v < 0.98 for v in eval_metrics["per_slice_viol_rate"]) and
              np.isfinite(eval_metrics.get("mean_return", np.nan)))

    # Save final checkpoint (periodic ones already in per_run_dir)
    ckpt_path = per_run_dir / f"{tag}_final.pt"
    try:
        trainer.save(str(ckpt_path))
    except Exception as e:
        print(f"  (final checkpoint save failed: {e})")

    # Per-run summary so each run_dir is self-describing
    run_summary = {
        "mode": mode, "template": template, "seed": seed,
        "status": status, "stable": bool(stable), "error": error,
        "train_time_sec": round(train_time, 1),
        "total_steps": total_steps,
        "n_updates": trainer.update_idx,
        "use_lagrangian": use_lagrangian,
        "normalize_per_slice_rewards": normalize_per_slice_rewards,
        "final_lambda": (trainer.lagrangian.lam.tolist() if trainer.lagrangian else None),
        "eval_summary": {k: v for k, v in eval_metrics.items()
                         if k not in ("trajectory_csv", "stochastic")},
        "artifacts": {
            "training_log_jsonl":  f"{tag}_training_log.jsonl",
            "training_log_json":   f"{tag}_training_log.json",
            "eval_trajectory_csv": f"{tag}_eval_trajectory.csv",
            "eval_stochastic_trajectory_csv": f"{tag}_eval_stochastic_trajectory.csv",
            "final_checkpoint":    f"{tag}_final.pt",
        },
    }
    (per_run_dir / f"{tag}_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=str), encoding="utf-8")

    return {
        "mode": mode,
        "template": template,
        "seed": seed,
        "status": status,
        "stable": bool(stable),
        "error": error,
        "train_time_sec": round(train_time, 1),
        "train_log_tail": train_log[-5:] if train_log else [],
        "eval": eval_metrics,
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
        "run_dir": str(per_run_dir),
    }


# ============================================================
# Decision logic
# ============================================================

def decide(atc_results: list, vanilla_results: list) -> dict:
    """Apply PROMOTE / FALLBACK / AMBIGUOUS criteria."""
    def agg(results, key_path):
        vals = []
        for r in results:
            if not r.get("stable"):
                continue
            v = r["eval"]
            for k in key_path:
                if not isinstance(v, dict):
                    v = None; break
                v = v.get(k)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            vals.append(v)
        return vals

    atc_stable = sum(1 for r in atc_results if r["stable"])
    van_stable = sum(1 for r in vanilla_results if r["stable"])
    n_atc, n_van = len(atc_results), len(vanilla_results)

    atc_most_c = agg(atc_results, ["most_constrained_slice_viol"])
    van_most_c = agg(vanilla_results, ["most_constrained_slice_viol"])
    atc_agg_u = agg(atc_results, ["aggregate_util"])
    van_agg_u = agg(vanilla_results, ["aggregate_util"])

    summary = {
        "atc_stable_seeds": f"{atc_stable}/{n_atc}",
        "vanilla_stable_seeds": f"{van_stable}/{n_van}",
        "atc_most_constrained_viol_mean": float(np.mean(atc_most_c)) if atc_most_c else None,
        "atc_most_constrained_viol_std":  float(np.std(atc_most_c))  if atc_most_c else None,
        "vanilla_most_constrained_viol_mean": float(np.mean(van_most_c)) if van_most_c else None,
        "vanilla_most_constrained_viol_std":  float(np.std(van_most_c))  if van_most_c else None,
        "atc_aggregate_u_mean":     float(np.mean(atc_agg_u)) if atc_agg_u else None,
        "vanilla_aggregate_u_mean": float(np.mean(van_agg_u)) if van_agg_u else None,
    }

    if atc_stable == 0 or van_stable == 0:
        summary["decision"] = "FALLBACK"
        summary["reason"] = f"insufficient stable seeds (ATC {atc_stable}/{n_atc}, Vanilla {van_stable}/{n_van})"
        return summary

    if atc_most_c and van_most_c and van_most_c[0] > 1e-6:
        ratio = np.mean(van_most_c) / max(np.mean(atc_most_c), 1e-6)
    else:
        ratio = None
    summary["most_constrained_viol_ratio_vanilla_over_atc"] = ratio

    # U criterion is one-sided: ATC is only penalized when it falls behind
    # Vanilla on U by more than the tolerance. Beating Vanilla (negative
    # agg_u_shortfall) is pure upside, since ATC should not sacrifice
    # utilization to reduce violations. Beating Vanilla on both U and the
    # violation criterion means ATC is strictly dominant.
    if atc_agg_u and van_agg_u:
        atc_u = float(np.mean(atc_agg_u))
        van_u = float(np.mean(van_agg_u))
        agg_u_shortfall = (van_u - atc_u) / max(van_u, 1e-6)   # +ve = ATC loses; -ve = ATC wins
    else:
        atc_u = van_u = None
        agg_u_shortfall = None
    summary["aggregate_u_shortfall_atc_vs_vanilla"] = agg_u_shortfall
    summary["aggregate_u_relative_gap"] = abs(agg_u_shortfall) if agg_u_shortfall is not None else None

    all_stable = (atc_stable / n_atc >= SUCCESS_CRITERIA["min_stable_seeds_fraction"] and
                  van_stable / n_van >= SUCCESS_CRITERIA["min_stable_seeds_fraction"])

    u_ok = (agg_u_shortfall is not None
            and agg_u_shortfall <= SUCCESS_CRITERIA["agg_u_tolerance"])

    if (all_stable and ratio is not None and ratio >= SUCCESS_CRITERIA["most_constrained_ratio"]
            and u_ok):
        summary["decision"] = "PROMOTE"
        if agg_u_shortfall < 0:
            u_story = f"aggregate U {-agg_u_shortfall*100:.1f}% HIGHER than Vanilla (strictly dominant)"
        else:
            u_story = f"aggregate U within {agg_u_shortfall*100:.1f}% of Vanilla"
        summary["reason"] = (f"all seeds stable; ATC most-constrained viol {ratio:.2f}x lower "
                             f"than Vanilla; {u_story}")
    elif (atc_stable / n_atc >= AMBIGUOUS_CRITERIA["min_stable_seeds_fraction"]
          and ratio is not None and ratio >= AMBIGUOUS_CRITERIA["most_constrained_ratio_min"]):
        summary["decision"] = "AMBIGUOUS"
        summary["reason"] = (f"partial success: ATC ratio {ratio:.2f}x (>=1.3 threshold), "
                             f"stable seeds {atc_stable}/{n_atc}; requires user judgment")
    else:
        summary["decision"] = "FALLBACK"
        bits = []
        if ratio is not None and ratio < AMBIGUOUS_CRITERIA["most_constrained_ratio_min"]:
            bits.append(f"ATC most-constrained viol ratio only {ratio:.2f}x (< 1.3)")
        if agg_u_shortfall is not None and agg_u_shortfall > SUCCESS_CRITERIA["agg_u_tolerance"]:
            bits.append(f"aggregate U {agg_u_shortfall*100:.1f}% BELOW Vanilla (> 10% shortfall)")
        if atc_stable / n_atc < SUCCESS_CRITERIA["min_stable_seeds_fraction"]:
            bits.append(f"only {atc_stable}/{n_atc} ATC seeds stable")
        summary["reason"] = "; ".join(bits) if bits else "criteria not met"

    return summary


# ============================================================
# Markdown report
# ============================================================

def _fmt(v, spec=".3f"):
    """Format-safely: missing / non-numeric -> '-'."""
    if v is None or isinstance(v, str):
        return "-"
    try:
        return f"{v:{spec}}"
    except (TypeError, ValueError):
        return "-"


def write_report(out_dir: Path, all_results: dict):
    md = []
    md.append(f"# K=3 Experiment Report — {all_results['timestamp']}")
    md.append("")
    md.append(f"Template: **{all_results['template']}**")
    md.append(f"Seeds: {all_results['seeds']}")
    md.append(f"Total steps per seed: {all_results['total_steps']}")
    md.append(f"Wall-clock time: {all_results['wall_clock_sec']:.0f}s "
              f"({all_results['wall_clock_sec']/60:.1f} min)")
    md.append("")

    md.append("## Decision summary")
    md.append("")
    dec = all_results["decision"]
    md.append(f"Decision: `{dec['decision']}`")
    md.append("")
    md.append(f"Reason: {dec['reason']}")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    for k, v in dec.items():
        if k in ("decision", "reason"): continue
        md.append(f"| {k} | {v} |")
    md.append("")

    md.append("## Per-seed ATC results")
    md.append("")
    md.append("| seed | stable | most-constr viol | agg U | per-slice viol | train(s) |")
    md.append("|---|---|---|---|---|---|")
    for r in all_results["atc_results"]:
        ev = r.get("eval", {})
        ps = ev.get("per_slice_viol_rate", [])
        psf = "[" + ", ".join(f"{v:.3f}" for v in ps) + "]" if ps else "-"
        md.append(f"| {r['seed']} | {'OK' if r['stable'] else 'unstable'} "
                  f"| {_fmt(ev.get('most_constrained_slice_viol'))} "
                  f"| {_fmt(ev.get('aggregate_util'))} "
                  f"| {psf} | {_fmt(r.get('train_time_sec'), '.1f')} |")
    md.append("")

    md.append("## Per-seed Vanilla-PPO results")
    md.append("")
    md.append("| seed | stable | most-constr viol | agg U | per-slice viol | train(s) |")
    md.append("|---|---|---|---|---|---|")
    for r in all_results["vanilla_results"]:
        ev = r.get("eval", {})
        ps = ev.get("per_slice_viol_rate", [])
        psf = "[" + ", ".join(f"{v:.3f}" for v in ps) + "]" if ps else "-"
        md.append(f"| {r['seed']} | {'OK' if r['stable'] else 'unstable'} "
                  f"| {_fmt(ev.get('most_constrained_slice_viol'))} "
                  f"| {_fmt(ev.get('aggregate_util'))} "
                  f"| {psf} | {_fmt(r.get('train_time_sec'), '.1f')} |")
    md.append("")

    # Extras (Oracle / SafeSlice / Static / etc.)
    for extra_mode, runs in (all_results.get("extra_results") or {}).items():
        md.append(f"## Per-seed {extra_mode} (reference only, not used in decision)")
        md.append("")
        md.append("| seed | stable | most-constr viol | agg U | per-slice viol | train(s) |")
        md.append("|---|---|---|---|---|---|")
        for r in runs:
            ev = r.get("eval", {})
            ps = ev.get("per_slice_viol_rate", [])
            psf = "[" + ", ".join(f"{v:.3f}" for v in ps) + "]" if ps else "-"
            md.append(f"| {r['seed']} | {'OK' if r['stable'] else 'unstable'} "
                      f"| {_fmt(ev.get('most_constrained_slice_viol'))} "
                      f"| {_fmt(ev.get('aggregate_util'))} "
                      f"| {psf} | {_fmt(r.get('train_time_sec'), '.1f')} |")
        md.append("")

    # Hyperparameter block so we can tell runs apart post-hoc
    if all_results.get("hyperparams"):
        md.append("## Hyperparameters")
        md.append("")
        md.append("| Key | Value |")
        md.append("|---|---|")
        for k, v in all_results["hyperparams"].items():
            md.append(f"| {k} | {v} |")
        md.append("")

    md.append("## Template definition (for record)")
    md.append("")
    md.append("```json")
    md.append(json.dumps(TEMPLATES[all_results["template"]], indent=2))
    md.append("```")

    (out_dir / "eval.md").write_text("\n".join(md), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", choices=list(TEMPLATES.keys()), default="A")
    parser.add_argument("--seeds", type=int, default=3, help="number of random seeds")
    parser.add_argument("--steps", type=int, default=60_000,
                        help="total_steps PER seed PER mode")
    parser.add_argument("--use-kalman", action="store_true",
                        help="use Kalman belief engine (default: discrete 3-state, matching K=1 ATC)")
    parser.add_argument("--modes", nargs="+", default=["proposed", "vanilla_ppo"],
                        choices=["proposed", "vanilla_ppo", "oracle", "safeslice",
                                 "static_slicing", "guardrail_only", "lstm_predictive"])
    parser.add_argument("--output", default=None, help="output dir (default: experiments/k3_<ts>)")
    parser.add_argument("--sanity-only", action="store_true",
                        help="run Template C first; abort if ATC not stable on >=2/3 seeds")
    parser.add_argument("--no-lagrangian", action="store_true", help="ablation: disable Fix 3b")
    parser.add_argument("--no-reward-norm", action="store_true", help="ablation: disable Fix 3a")
    parser.add_argument("--lr-dual", type=float, default=1e-3,
                        help="per-slice Lagrangian learning rate (default 1e-3). Higher -> stronger pressure on violating slice.")
    parser.add_argument("--alpha-floor", type=float, default=1.0,
                        help="Dirichlet alpha floor (default 1.0 = uniform start). <1.0 -> sharper, >1.0 -> more uniform.")
    parser.add_argument("--sla-budget", type=str, default=None,
                        help="override per-slice violation budget as comma-separated list, e.g. '0.03,0.05,0.10'. If unset, use template default.")
    args = parser.parse_args()

    sla_budget_override = None
    if args.sla_budget:
        try:
            sla_budget_override = [float(x) for x in args.sla_budget.split(",")]
            assert len(sla_budget_override) == 3
        except Exception as e:
            print(f"!!! Bad --sla-budget '{args.sla_budget}': {e}")
            return

    # Output dir
    tz = timezone(timedelta(hours=9))  # Asia/Seoul-like; edit for your server
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path(f"experiments/k3_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Save run config
    run_cfg = {
        "timestamp": ts,
        "args": vars(args),
        "templates": TEMPLATES,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    # Optionally run Template-C sanity only (explicit --sanity-only).
    # Running --template C as a normal decision run does NOT trigger this.
    if args.sanity_only:
        print("\n" + "=" * 60)
        print(">>> SANITY-ONLY (Template C, identical slices, half steps)")
        print("=" * 60)
        sanity_atc = [
            run_one("proposed", "C", seed, args.steps // 2,
                    args.use_kalman, out_dir / "sanity",
                    use_lagrangian=not args.no_lagrangian,
                    normalize_per_slice_rewards=not args.no_reward_norm,
                    lr_dual=args.lr_dual, alpha_floor=args.alpha_floor,
                    sla_budget_override=sla_budget_override)
            for seed in range(args.seeds)
        ]
        ok = sum(1 for r in sanity_atc if r["stable"])
        print(f"\nSANITY: ATC stable {ok}/{args.seeds}")
        (out_dir / "sanity_results.json").write_text(
            json.dumps(sanity_atc, indent=2, default=str), encoding="utf-8")
        if ok < 2:
            print("!!! SANITY FAILED — investigate before running Template A")
        return

    # Main decision run (Template A or whatever user picked)
    print("\n" + "=" * 60)
    print(f">>> DECISION RUN (Template {args.template}, heterogeneous)")
    print("=" * 60)

    t0 = time.time()
    atc_results = []
    vanilla_results = []
    if "proposed" in args.modes:
        for seed in range(args.seeds):
            atc_results.append(run_one("proposed", args.template, seed, args.steps,
                                       args.use_kalman, out_dir,
                                       use_lagrangian=not args.no_lagrangian,
                                       normalize_per_slice_rewards=not args.no_reward_norm,
                                       lr_dual=args.lr_dual,
                                       alpha_floor=args.alpha_floor,
                                       sla_budget_override=sla_budget_override))
    if "vanilla_ppo" in args.modes:
        for seed in range(args.seeds):
            vanilla_results.append(run_one("vanilla_ppo", args.template, seed, args.steps,
                                           args.use_kalman, out_dir,
                                           # vanilla should NOT use lagrangian/reward-norm,
                                           # so the comparison is clean (ATC = all-three-fixes;
                                           # vanilla = no fixes + Dirichlet only)
                                           use_lagrangian=False,
                                           normalize_per_slice_rewards=False,
                                           alpha_floor=args.alpha_floor,
                                           sla_budget_override=sla_budget_override))
    # Extras (oracle / safeslice / static_slicing etc.) are NOT part of the binary
    # ATC-vs-Vanilla decision, but we still train+save them as reference baselines.
    extra_results_by_mode = {}
    for extra_mode in args.modes:
        if extra_mode in ("proposed", "vanilla_ppo"):
            continue
        runs = []
        for seed in range(args.seeds):
            runs.append(run_one(extra_mode, args.template, seed, args.steps,
                                args.use_kalman, out_dir,
                                use_lagrangian=False,
                                normalize_per_slice_rewards=False,
                                alpha_floor=args.alpha_floor,
                                sla_budget_override=sla_budget_override))
        extra_results_by_mode[extra_mode] = runs

    decision = decide(atc_results, vanilla_results)

    wall = time.time() - t0
    all_results = {
        "timestamp": ts,
        "template": args.template,
        "seeds": list(range(args.seeds)),
        "total_steps": args.steps,
        "wall_clock_sec": wall,
        "atc_results": atc_results,
        "vanilla_results": vanilla_results,
        "extra_results": extra_results_by_mode,
        "decision": decision,
        "hyperparams": {
            "lr_dual": args.lr_dual,
            "alpha_floor": args.alpha_floor,
            "use_lagrangian": not args.no_lagrangian,
            "normalize_per_slice_rewards": not args.no_reward_norm,
            "sla_budget": sla_budget_override,
            "use_kalman": args.use_kalman,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2, default=str),
                                          encoding="utf-8")
    write_report(out_dir, all_results)

    print("\n" + "=" * 60)
    print(f">>> DECISION: {decision['decision']}")
    print(f"    Reason: {decision['reason']}")
    print(f"    Wall time: {wall/60:.1f} min")
    print(f"    Eval:     {out_dir / 'eval.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
