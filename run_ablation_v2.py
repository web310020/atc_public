"""
Ablation 实验 (K-slice 通用).

对比: ATC (完整版) / 去掉 belief engine / 去掉 trust fusion.
输出 ablation_results_v2.md.

Usage:
    python run_ablation_v2.py
"""

import os
import glob
import json
import numpy as np
import time
from scipy.stats import pearsonr
from stable_baselines3 import PPO
from core.telemetry_env import E2_Node_Simulator

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ATC_NoBelief(E2_Node_Simulator):
    """ATC w/o Belief Engine: freeze all belief engines at uniform prior."""
    def step(self, xapp_action):
        # Freeze all belief engines before step
        for be in self.belief_engines:
            be.reset()  # Reset to uniform prior each step
        # Run normal step (belief will be at max entropy, gamma ≈ 0)
        return super().step(xapp_action)


class ATC_NoTrust(E2_Node_Simulator):
    """ATC w/o Trust Fusion: force gamma=1 (pure RIC, no BS blending)."""
    def step(self, xapp_action):
        # Normal step but override the mode logic to skip trust fusion
        if np.isscalar(xapp_action) or len(xapp_action) == 1:
            raw_action = np.full(self.K, float(xapp_action[0]) if not np.isscalar(xapp_action) else float(xapp_action))
        else:
            raw_action = np.array(xapp_action[:self.K], dtype=np.float64)

        # Belief + telemetry run normally
        for k in range(self.K):
            self.telemetry_controllers[k].adapt(self.belief_engines[k], tau=self.tau_per[k])

        # RIC-side safety fusion runs normally
        ric_action = np.zeros(self.K)
        current_lam = np.zeros(self.K)
        for k in range(self.K):
            ric_action[k], current_lam[k] = self.security_unit.verify_and_fuse(
                raw_action[k], self.belief_engines[k]
            )
        ric_action = self._simplex_project(ric_action)

        # KEY: force gamma=1, skip BS guardrail
        gamma = 1.0
        final_action = ric_action  # Pure RIC, no local blending

        # Physics
        for k in range(self.K):
            traffic_flux = np.random.normal(0, self.sigma_k[k])
            if self.K == 1:
                action_effect = (0.5 - final_action[k]) * 0.1
                drift = 0.02
            else:
                neutral = 1.0 / self.K
                action_effect = (neutral - final_action[k]) * 0.1 * self.K
                drift = 0.01
            self.true_util[k] = np.clip(
                0.95 * self.true_util[k] + action_effect + traffic_flux + drift, 0, 1.0
            )

        # Sampling + belief update
        total_sig_cost = 0.0
        sampled_any = False
        for k in range(self.K):
            self.steps_since_last_indication[k] += self.sim_step_ms
            if self.steps_since_last_indication[k] >= self.telemetry_controllers[k].period:
                self.last_reported_kpm[k] = self.true_util[k]
                self.belief_engines[k].update(self.last_reported_kpm[k])
                self.steps_since_last_indication[k] = 0
                total_sig_cost += 1.0
                sampled_any = True
            else:
                self.belief_engines[k].predict(final_action[k], K=self.K)

        # Reward
        r_sla, r_perf = 0.0, 0.0
        for k in range(self.K):
            if self.true_util[k] > self.tau_k[k]:
                r_sla += -self.beta * (self.true_util[k] - self.tau_k[k])
            diff = self.true_util[k] - self.target_prb_k[k]
            r_perf += -(diff ** 2) * (40.0 if diff > 0 else 4.0)
        r_cost = -2.0 * total_sig_cost if (sampled_any and self.mode != "oracle") else 0.0
        total_reward = r_perf + r_sla + r_cost

        # Info
        avg_util = np.mean(self.true_util)
        avg_belief = np.mean([be.get_mean() for be in self.belief_engines])
        entropies = np.array([be.get_entropy() for be in self.belief_engines])
        info = {
            "true_util": avg_util,
            "belief_mean": avg_belief,
            "entropy": np.mean(entropies),
            "lambda": np.mean(current_lam),
            "gamma": gamma,
            "period": np.mean([tc.period for tc in self.telemetry_controllers]),
            "sig_cost": total_sig_cost,
            "is_violation": float(np.any(self.true_util > self.tau_k)),
        }
        self.history.append(info)

        # Obs
        belief_means = np.array([be.get_mean() for be in self.belief_engines])
        entropies_now = np.array([be.get_entropy() for be in self.belief_engines])
        obs_vec = np.concatenate([
            [np.mean(self.last_reported_kpm), np.mean(self.tau_k)],
            belief_means, entropies_now, self.tau_k
        ]).astype(np.float32)

        return obs_vec, total_reward, False, False, info


def evaluate_single(env, model, n_steps=1000):
    stats = {"util": [], "viol_depth": [], "viol": [], "cost": [], "reward": []}
    obs, _ = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(action)
        u = info["true_util"]
        stats["util"].append(u)
        stats["viol_depth"].append(max(0, u - 0.5))
        stats["viol"].append(info["is_violation"])
        stats["cost"].append(info["sig_cost"])
        stats["reward"].append(reward)
    return {
        "U": np.mean(stats["util"]),
        "delta": np.mean(stats["viol_depth"]),
        "viol_pct": np.mean(stats["viol"]) * 100,
        "sig_cost": np.sum(stats["cost"]),
        "psi": abs(np.mean(stats["reward"])),
    }


def run_ablation(n_runs=50, n_steps=1000):
    # Find latest experiment
    exps = glob.glob(os.path.join("experiments", "*"))
    latest = max(exps, key=os.path.getmtime)
    model_path = os.path.join(latest, "proposed", "models", "ppo_proposed_final")
    if not os.path.exists(model_path + ".zip"):
        print(f"ERROR: Model not found at {model_path}.zip")
        return

    config_path = os.path.join(latest, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        K = cfg.get("K", 1)
        use_kalman = cfg.get("use_kalman", False)
    else:
        K, use_kalman = 1, False

    model = PPO.load(model_path, device='cpu')
    print(f"Loaded: {model_path} (K={K}, KF={'ON' if use_kalman else 'OFF'})")
    print(f"Running {n_runs} runs × {n_steps} steps × 3 variants...\n")

    variants = {
        "ATC (Full)": lambda: E2_Node_Simulator(mode="proposed", K=K, use_kalman=use_kalman),
        "w/o Belief Engine": lambda: ATC_NoBelief(mode="proposed", K=K, use_kalman=use_kalman),
        "w/o Trust Fusion": lambda: ATC_NoTrust(mode="proposed", K=K, use_kalman=use_kalman),
    }

    all_results = {}
    for name, factory in variants.items():
        print(f"  {name} ...", end="", flush=True)
        runs = []
        for _ in range(n_runs):
            env = factory()
            env.set_a1_policy(0.5)
            runs.append(evaluate_single(env, model, n_steps))
        agg = {}
        for key in runs[0]:
            vals = [r[key] for r in runs]
            agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}
        all_results[name] = agg
        print(f" U={agg['U']['mean']:.3f}±{agg['U']['std']:.3f}, "
              f"δ={agg['delta']['mean']:.4f}, V={agg['viol_pct']['mean']:.1f}%")

    lines = [
        "# Ablation Results v2",
        f"**Experiment:** `{latest}`",
        f"**Protocol:** {n_runs} runs × {n_steps} steps",
        "",
        "| Variant | U ↑ | δ ↓ | Viol.% ↓ | Sig | Feasible (≤5%)? |",
        "|---------|-----|-----|----------|-----|-----------------|",
    ]
    for name, agg in all_results.items():
        feasible = "yes" if agg['viol_pct']['mean'] <= 5.0 else "no"
        lines.append(
            f"| **{name}** | {agg['U']['mean']:.3f}±{agg['U']['std']:.3f} | "
            f"{agg['delta']['mean']:.4f}±{agg['delta']['std']:.4f} | "
            f"{agg['viol_pct']['mean']:.1f}±{agg['viol_pct']['std']:.1f}% | "
            f"{agg['sig_cost']['mean']:.0f} | {feasible} |"
        )
    lines.extend(["", "---", "*Generated by run_ablation_v2.py*"])

    out_path = os.path.join(latest, "ablation_results_v2.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n>>> Saved: {out_path}")


if __name__ == "__main__":
    run_ablation(n_runs=50, n_steps=1000)
