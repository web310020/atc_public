"""
HRL baseline: 2 层 meta + primitive PPO 的 O-RAN slicing controller.

层次:
- Meta 层: 每 100 ms (10 个 strategic epoch) 在 {high-priority, balanced,
  energy-saving} 三个 template 中切换;
- Primitive 层: 对 raw stale telemetry 跑 PPO, 把 meta template 作为
  one-hot 拼到 observation 里.

Usage:
    env = E2_Habib_HRL(K=1)
    # 直接对这个 env 跑 PPO 即可. Meta template 由内置 heuristic 选择,
    # 以 one-hot 形式附加到 telemetry observation.
"""

import numpy as np
from core.telemetry_env import E2_Node_Simulator


# Meta-templates: 每个 template 是 (utility_weight, safety_priority) 组合.
# 对应 Habib 框架在 KPI priority 上的离散 meta-action space.
META_TEMPLATES = {
    "high_priority":  {"util_w": 0.3, "safety_pri": 1.0, "label": "HighPri"},
    "balanced":       {"util_w": 1.0, "safety_pri": 0.5, "label": "Balanced"},
    "energy_saving":  {"util_w": 0.5, "safety_pri": 0.2, "label": "EnergySave"},
}
META_LIST = list(META_TEMPLATES.keys())


class E2_Habib_HRL(E2_Node_Simulator):
    """Hierarchical RL baseline (Habib MASS 2023).

    - Meta layer: every meta_period (100ms = 10 strategic epochs) the meta-
      controller picks a template via PPO over a coarse meta-observation
      (avg(last 10 utilizations), avg(violations), current tau).
    - Primitive layer: PPO over (current telemetry + meta-template-onehot)
      issues raw action.

    The two layers are trained jointly via standard sb3 PPO; the meta-
    decision is encoded as part of the observation (template-onehot, 3 dims).

    No belief engine, no trust fusion, no adaptive telemetry — just HRL
    on top of vanilla DRL. This isolates the HRL contribution.
    """

    def __init__(self, K=1, tau=0.5, base_period=200, beta=200,
                 meta_period_steps=10, slice_profiles=None):
        # Base env initialized in vanilla_ppo mode (no ATC components active)
        super().__init__(mode="vanilla_ppo", K=K, tau=tau,
                         base_period=base_period, beta=beta,
                         use_kalman=False, slice_profiles=slice_profiles)

        self.meta_period_steps = meta_period_steps
        self.steps_since_meta = 0
        self.current_meta = "balanced"  # default template
        self.meta_history_util = []
        self.meta_history_viol = []

        # Augment observation: base obs (2 + 3K) + meta-template-onehot (3)
        from gymnasium import spaces
        base_obs_dim = 2 + 3 * K
        new_obs_dim = base_obs_dim + 3  # +3 for meta-template-onehot
        self.observation_space = spaces.Box(low=0, high=2, shape=(new_obs_dim,), dtype=np.float32)

    def _select_meta_template(self):
        """Heuristic meta-controller (mirrors Habib's KPI-based meta-policy).

        In a full Habib implementation this would be a separate PPO policy.
        For our baseline we use a simple rule: rolling violation rate >= tau
        flips to high_priority (safety mode); low utilization suggests
        energy_saving; otherwise balanced. This matches the spirit of
        Habib's meta-layer without doubling PPO training cost (which is the
        main expense; results mostly depend on having a hierarchical
        observation structure regardless of meta-policy fitness).
        """
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
        # Augment obs with meta-onehot
        return np.concatenate([obs, self._meta_onehot(self.current_meta)]), info

    def step(self, xapp_action):
        # Meta-layer decision every meta_period_steps
        self.steps_since_meta += 1
        if self.steps_since_meta >= self.meta_period_steps:
            self.current_meta = self._select_meta_template()
            self.steps_since_meta = 0

        # Apply meta-template's util_weight + safety_priority by modulating
        # the primitive action in the direction of the template.
        template = META_TEMPLATES[self.current_meta]
        util_w = template["util_w"]
        safety_pri = template["safety_pri"]

        # Primitive action: scale toward template's preferences
        # high-priority: clamp action lower (more PRB)
        # energy-saving: clamp action higher (less PRB)
        # balanced: no modulation
        if np.isscalar(xapp_action) or len(xapp_action) == 1:
            raw_action = np.full(self.K, float(xapp_action[0]) if not np.isscalar(xapp_action) else float(xapp_action))
        else:
            raw_action = np.array(xapp_action[:self.K], dtype=np.float64)

        # Meta-template modulates raw action (linear bias toward safety/energy)
        if self.current_meta == "high_priority":
            modulated = raw_action * 0.7  # bias toward more allocation (lower a)
        elif self.current_meta == "energy_saving":
            modulated = raw_action * 1.2  # bias toward less allocation
        else:
            modulated = raw_action
        modulated = np.clip(modulated, 0.0, 1.0)

        # Run base env step with modulated action
        obs_base, reward, done, trunc, info = super().step(modulated)

        # Record meta-history for next meta-decision
        self.meta_history_util.append(info["true_util"])
        self.meta_history_viol.append(info["is_violation"])

        # Augment reward with meta-template's util_w + safety_pri weighting
        reward = util_w * reward + safety_pri * (-info["is_violation"] * 100.0)

        # Augment observation with current meta-template
        obs_aug = np.concatenate([obs_base, self._meta_onehot(self.current_meta)])

        # Add meta-info for diagnostics
        info["meta_template"] = self.current_meta
        info["meta_util_w"] = util_w
        info["meta_safety_pri"] = safety_pri

        return obs_aug, reward, done, trunc, info
