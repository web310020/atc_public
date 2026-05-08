# core/k3_env.py
# ============================================================
# K=3 Environment Wrapper for the Belief-Telemetry Experiment
# ============================================================
# Thin subclass of E2_Node_Simulator that:
#   - sets heterogeneous per-slice SLA thresholds (tau_k) per template
#   - exposes per-slice reward in info["per_slice_reward"]  (needed by Fix 3)
#   - keeps per-slice violation info already provided by the parent env
# The actual physics, belief engine, trust fusion, simplex projection, and
# guardrail are ALL inherited unchanged from the validated K=1 env.
# ============================================================

import numpy as np
from core.telemetry_env import E2_Node_Simulator, SLICE_PROFILES


# Per-template slice configuration.
#   tau:        SLA violation threshold (utilization >=tau counts as violation)
#   target_prb: target utilization (reward-shaping center)
#   base_util:  episode-start utilization
#   sigma:      per-step traffic noise
#   sla_budget: per-slice acceptable long-run violation rate (for Lagrangian)
TEMPLATES = {
    # Template C: 3 个相同 slice, sanity check 用 (K=1 复制 3 次)
    "C": {
        "names":      ["SliceA",  "SliceB",  "SliceC"],
        "tau":        [0.50,      0.50,      0.50],
        "target_prb": [0.40,      0.40,      0.40],
        "base_util":  [0.35,      0.35,      0.35],
        "sigma":      [0.05,      0.05,      0.05],
        "sla_budget": [0.05,      0.05,      0.05],
    },
    # Template A: 异构 URLLC / eMBB / mMTC, 主测试用
    # URLLC is the MOST-CONSTRAINED slice; we report its violation rate separately.
    "A": {
        "names":      ["URLLC",   "eMBB",    "mMTC"],
        "tau":        [0.35,      0.50,      0.60],
        "target_prb": [0.30,      0.45,      0.55],
        "base_util":  [0.25,      0.35,      0.40],
        "sigma":      [0.03,      0.05,      0.08],
        "sla_budget": [0.01,      0.05,      0.10],
    },
    # Template B: Safety / Normal / Background, 另一种异构组合
    "B": {
        "names":      ["Safety",  "Normal",  "Background"],
        "tau":        [0.30,      0.55,      0.70],
        "target_prb": [0.25,      0.50,      0.60],
        "base_util":  [0.20,      0.40,      0.45],
        "sigma":      [0.02,      0.05,      0.10],
        "sla_budget": [0.005,     0.05,      0.15],
    },
    # Template K5_A: K=5 异构 slice 组合.
    # URLLC (最紧) + V2X-safety + eMBB + mMTC + IoT-bursty (最宽松)
    "K5_A": {
        "names":      ["URLLC",   "V2X",     "eMBB",    "mMTC",    "IoT_burst"],
        "tau":        [0.30,      0.35,      0.50,      0.60,      0.65],
        "target_prb": [0.25,      0.28,      0.45,      0.55,      0.60],
        "base_util":  [0.20,      0.23,      0.35,      0.40,      0.45],
        "sigma":      [0.02,      0.03,      0.05,      0.08,      0.10],
        "sla_budget": [0.005,     0.01,      0.05,      0.10,      0.15],
    },
}


class E2_Node_K3_Env(E2_Node_Simulator):
    """
    K=3 drop-in for E2_Node_Simulator with per-template heterogeneous SLA
    and per-slice reward exposure in the info dict.

    `mode` is forwarded unchanged ('proposed', 'vanilla_ppo', 'oracle', etc.)
    """
    def __init__(self, mode="proposed", template="A", use_kalman=False,
                 base_period=200, beta=200):
        assert template in TEMPLATES, f"unknown template: {template}"
        cfg = TEMPLATES[template]
        K = len(cfg["names"])  # generalized: 3 for A/B/C, 5 for K5_A, etc.
        # Build slice_profiles dict so parent picks them up
        slice_profiles = {
            cfg["names"][k]: {
                "sigma":      cfg["sigma"][k],
                "base_util":  cfg["base_util"][k],
                "target_prb": cfg["target_prb"][k],
                "label":      cfg["names"][k],
            }
            for k in range(K)
        }
        # Temporarily inject into SLICE_PROFILES for parent __init__ to find them
        _saved = {n: SLICE_PROFILES.get(n) for n in cfg["names"]}
        for n, p in slice_profiles.items():
            SLICE_PROFILES[n] = p

        super().__init__(
            mode=mode,
            K=K,
            tau=np.mean(cfg["tau"]),   # parent uses scalar; we overwrite below
            base_period=base_period,
            beta=beta,
            use_kalman=use_kalman,
            slice_profiles=slice_profiles,
        )

        # Restore any keys we temporarily overrode (so subsequent envs see originals)
        for n, old in _saved.items():
            if old is None:
                SLICE_PROFILES.pop(n, None)
            else:
                SLICE_PROFILES[n] = old

        # Override parent's homogeneous tau_k with the template's heterogeneous one
        self.tau_k = np.asarray(cfg["tau"], dtype=np.float64)
        self.target_prb_k = np.asarray(cfg["target_prb"], dtype=np.float64)
        self.base_util_k = np.asarray(cfg["base_util"], dtype=np.float64)
        self.sigma_k = np.asarray(cfg["sigma"], dtype=np.float64)
        self.sla_budget = np.asarray(cfg["sla_budget"], dtype=np.float64)
        self.slice_names = cfg["names"]
        self.template = template

        # Reset to apply new base utilizations
        self.true_util = np.copy(self.base_util_k)
        self.last_reported_kpm = np.copy(self.base_util_k)

    # ------------------------------------------------------------
    def _compute_per_slice_reward(self):
        """Break down the aggregate reward (Eq.7) into per-slice components."""
        per_slice = np.zeros(self.K)
        for k in range(self.K):
            u_k = self.true_util[k]
            tau_k = self.tau_k[k]
            if u_k > tau_k:
                per_slice[k] += -self.beta * (u_k - tau_k)
            diff = u_k - self.target_prb_k[k]
            if diff > 0:
                per_slice[k] += -(diff ** 2) * 40.0
            else:
                per_slice[k] += -(diff ** 2) * 4.0
        return per_slice

    # ------------------------------------------------------------
    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        # Attach per-slice reward for Fix 3 (per-slice Lagrangian + reward norm)
        info["per_slice_reward"] = self._compute_per_slice_reward()
        info["slice_names"] = self.slice_names
        info["template"] = self.template
        info["tau_k"] = self.tau_k.copy()
        info["sla_budget"] = self.sla_budget.copy()
        return obs, reward, done, trunc, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["slice_names"] = self.slice_names
        info["template"] = self.template
        return obs, info


def make_env(mode: str, template: str = "A", use_kalman: bool = False):
    """Factory function used by the PPO trainer."""
    def _init(seed=0):
        env = E2_Node_K3_Env(mode=mode, template=template, use_kalman=use_kalman)
        env.reset(seed=seed)
        return env
    return _init
