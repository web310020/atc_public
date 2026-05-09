# core/k3_env.py
# ============================================================
# K=3 (and K=5) env wrapper. 在 E2_Node_Simulator 上叠加:
#   - 异质 per-slice tau_k (按 template)
#   - info["per_slice_reward"] 暴露 per-slice 分量 (per-slice Lagrangian 用)
# physics / belief / trust fusion / guardrail 全继承 K=1 env 不动.
# ============================================================

import numpy as np
from core.telemetry_env import E2_Node_Simulator, SLICE_PROFILES


# tau / target_prb / base_util / sigma / sla_budget per slice.
TEMPLATES = {
    # 同质 sanity check
    "C": {
        "names":      ["SliceA",  "SliceB",  "SliceC"],
        "tau":        [0.50,      0.50,      0.50],
        "target_prb": [0.40,      0.40,      0.40],
        "base_util":  [0.35,      0.35,      0.35],
        "sigma":      [0.05,      0.05,      0.05],
        "sla_budget": [0.05,      0.05,      0.05],
    },
    # 异质 URLLC / eMBB / mMTC (主要 test case, URLLC 最紧)
    "A": {
        "names":      ["URLLC",   "eMBB",    "mMTC"],
        "tau":        [0.35,      0.50,      0.60],
        "target_prb": [0.30,      0.45,      0.55],
        "base_util":  [0.25,      0.35,      0.40],
        "sigma":      [0.03,      0.05,      0.08],
        "sla_budget": [0.01,      0.05,      0.10],
    },
    # 异质 Safety / Normal / Background
    "B": {
        "names":      ["Safety",  "Normal",  "Background"],
        "tau":        [0.30,      0.55,      0.70],
        "target_prb": [0.25,      0.50,      0.60],
        "base_util":  [0.20,      0.40,      0.45],
        "sigma":      [0.02,      0.05,      0.10],
        "sla_budget": [0.005,     0.05,      0.15],
    },
    # K=5 异质: URLLC / V2X / eMBB / mMTC / IoT-bursty
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
    """K=3/K=5 drop-in env: per-template 异质 SLA + per-slice reward 暴露."""
    def __init__(self, mode="proposed", template="A", use_kalman=False,
                 base_period=200, beta=200):
        assert template in TEMPLATES, f"unknown template: {template}"
        cfg = TEMPLATES[template]
        K = len(cfg["names"])  # 3 (A/B/C) 或 5 (K5_A)
        slice_profiles = {
            cfg["names"][k]: {
                "sigma":      cfg["sigma"][k],
                "base_util":  cfg["base_util"][k],
                "target_prb": cfg["target_prb"][k],
                "label":      cfg["names"][k],
            }
            for k in range(K)
        }
        # 临时注入 SLICE_PROFILES 让 parent __init__ 看到, 之后 restore
        _saved = {n: SLICE_PROFILES.get(n) for n in cfg["names"]}
        for n, p in slice_profiles.items():
            SLICE_PROFILES[n] = p

        super().__init__(
            mode=mode,
            K=K,
            tau=np.mean(cfg["tau"]),
            base_period=base_period,
            beta=beta,
            use_kalman=use_kalman,
            slice_profiles=slice_profiles,
        )

        for n, old in _saved.items():
            if old is None:
                SLICE_PROFILES.pop(n, None)
            else:
                SLICE_PROFILES[n] = old

        # 用 template 的异质 tau_k / target_prb / base_util / sigma 覆盖 parent 的 homogeneous 设置
        self.tau_k = np.asarray(cfg["tau"], dtype=np.float64)
        self.target_prb_k = np.asarray(cfg["target_prb"], dtype=np.float64)
        self.base_util_k = np.asarray(cfg["base_util"], dtype=np.float64)
        self.sigma_k = np.asarray(cfg["sigma"], dtype=np.float64)
        self.sla_budget = np.asarray(cfg["sla_budget"], dtype=np.float64)
        self.slice_names = cfg["names"]
        self.template = template

        self.true_util = np.copy(self.base_util_k)
        self.last_reported_kpm = np.copy(self.base_util_k)

    def _compute_per_slice_reward(self):
        # Aggregate reward 拆 per-slice 分量
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

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
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
    # PPO trainer 用的 env factory
    def _init(seed=0):
        env = E2_Node_K3_Env(mode=mode, template=template, use_kalman=use_kalman)
        env.reset(seed=seed)
        return env
    return _init
