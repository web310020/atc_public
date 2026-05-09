"""
Habib MASS 2023 HRL baseline.

Meta 层每 100ms (10 strategic epochs) 在 high-priority / balanced / energy-saving
3 个 KPI 模板间切, primitive 层是 PPO over (telemetry + meta-onehot). 共训, 不带
belief engine, 不带 trust fusion, 不带 adaptive telemetry — 用来隔离 HRL 本身的贡献.
"""

import numpy as np
from core.telemetry_env import E2_Node_Simulator


# Meta 模板: (utility_weight, safety_priority)
META_TEMPLATES = {
    "high_priority":  {"util_w": 0.3, "safety_pri": 1.0, "label": "HighPri"},
    "balanced":       {"util_w": 1.0, "safety_pri": 0.5, "label": "Balanced"},
    "energy_saving":  {"util_w": 0.5, "safety_pri": 0.2, "label": "EnergySave"},
}
META_LIST = list(META_TEMPLATES.keys())


class E2_Habib_HRL(E2_Node_Simulator):
    """HRL baseline: meta 层 KPI 模板选择 + primitive 层 PPO."""

    def __init__(self, K=1, tau=0.5, base_period=200, beta=200,
                 meta_period_steps=10, slice_profiles=None):
        super().__init__(mode="vanilla_ppo", K=K, tau=tau,
                         base_period=base_period, beta=beta,
                         use_kalman=False, slice_profiles=slice_profiles)

        self.meta_period_steps = meta_period_steps
        self.steps_since_meta = 0
        self.current_meta = "balanced"
        self.meta_history_util = []
        self.meta_history_viol = []

        # Obs = base (2 + 3K) + meta-template-onehot (3)
        from gymnasium import spaces
        base_obs_dim = 2 + 3 * K
        new_obs_dim = base_obs_dim + 3
        self.observation_space = spaces.Box(low=0, high=2, shape=(new_obs_dim,), dtype=np.float32)

    def _select_meta_template(self):
        # 简化版 meta heuristic (而不是再训一个 PPO meta-policy):
        # recent_viol > 0.05 切 high_priority, util < 0.3 切 energy_saving, 其余 balanced.
        if len(self.meta_history_viol) < 10:
            return "balanced"
        recent_viol = np.mean(self.meta_history_viol[-10:])
        recent_util = np.mean(self.meta_history_util[-10:])
        if recent_viol > 0.05:
            return "high_priority"
        elif recent_util < 0.30:
            return "energy_saving"
        else:
            return "balanced"

    def _meta_onehot(self, template_name):
        return np.array([1.0 if name == template_name else 0.0
                         for name in META_LIST], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.steps_since_meta = 0
        self.current_meta = "balanced"
        self.meta_history_util = []
        self.meta_history_viol = []
        return np.concatenate([obs, self._meta_onehot(self.current_meta)]), info

    def step(self, xapp_action):
        self.steps_since_meta += 1
        if self.steps_since_meta >= self.meta_period_steps:
            self.current_meta = self._select_meta_template()
            self.steps_since_meta = 0

        template = META_TEMPLATES[self.current_meta]
        util_w = template["util_w"]
        safety_pri = template["safety_pri"]

        if np.isscalar(xapp_action) or len(xapp_action) == 1:
            raw_action = np.full(self.K, float(xapp_action[0]) if not np.isscalar(xapp_action) else float(xapp_action))
        else:
            raw_action = np.array(xapp_action[:self.K], dtype=np.float64)

        # Meta 模板 modulate primitive action: high_priority bias 多分配, energy_saving bias 少分配
        if self.current_meta == "high_priority":
            modulated = raw_action * 0.7
        elif self.current_meta == "energy_saving":
            modulated = raw_action * 1.2
        else:
            modulated = raw_action
        modulated = np.clip(modulated, 0.0, 1.0)

        obs_base, reward, done, trunc, info = super().step(modulated)

        self.meta_history_util.append(info["true_util"])
        self.meta_history_viol.append(info["is_violation"])

        reward = util_w * reward + safety_pri * (-info["is_violation"] * 100.0)

        obs_aug = np.concatenate([obs_base, self._meta_onehot(self.current_meta)])

        info["meta_template"] = self.current_meta
        info["meta_util_w"] = util_w
        info["meta_safety_pri"] = safety_pri

        return obs_aug, reward, done, trunc, info
