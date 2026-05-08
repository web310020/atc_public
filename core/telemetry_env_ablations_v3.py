"""
ATC 各组件 ablation 子类.

四种变体, 每个 ablation 只关掉/换掉一个组件:
- ATC_NoAdaptiveTelemetry: 固定 signaling 周期 (没有 entropy gating)
- ATC_NoSCU:               跳过 safety calibration unit (raw action 直接走到 reflex)
- ATC_BangBang:            binary guardrail (代替 proportional reflex)
- (use_kalman=True):       连续 belief 版本 (base simulator 自带的 flag)

每个子类只重写 step() 的关键路径, 其他逻辑跟 proposed ATC 完全一致,
这样可以单独看每个组件的贡献.

Usage:
    env = ATC_NoAdaptiveTelemetry(mode="proposed", K=1, ...)
    env = ATC_NoSCU(mode="proposed", K=1, ...)
    env = ATC_BangBang(mode="proposed", K=1, ...)
    env = E2_Node_Simulator(mode="proposed", K=1, use_kalman=True)
"""

import numpy as np
from core.telemetry_env import E2_Node_Simulator


# ============================================================
# Ablation 1: w/o entropy-driven adaptive telemetry
# ============================================================
class ATC_NoAdaptiveTelemetry(E2_Node_Simulator):
    """ATC 但 telemetry 周期固定 (没有 entropy-gated adaptation).

    把 adaptive_telemetry_controller.adapt() 关掉, telemetry 周期始终
    保持在 base_period (belief entropy 飙升时也不触发高频采样).
    用来看 entropy-driven gating 对 signaling efficiency 的实际贡献.

    预期: signaling cost 要么 INCREASE (没有 adaptive throttle)
    要么 DECREASE (没有 adaptive boost), 方向告诉我们 gating 的作用.
    """
    def step(self, xapp_action):
        # Force fixed period (skip adaptive logic)
        if np.isscalar(xapp_action) or len(xapp_action) == 1:
            raw_action = np.full(self.K, float(xapp_action[0]) if not np.isscalar(xapp_action) else float(xapp_action))
        else:
            raw_action = np.array(xapp_action[:self.K], dtype=np.float64)

        entropies = np.array([be.get_entropy() for be in self.belief_engines])

        # Skip adaptive telemetry — keep period constant
        for tc in self.telemetry_controllers:
            tc.period = self.base_period

        # Rest of "proposed" logic preserved
        ric_action = np.zeros(self.K)
        current_lam = np.zeros(self.K)
        for k in range(self.K):
            ric_action[k], current_lam[k] = self.security_unit.verify_and_fuse(
                raw_action[k], self.belief_engines[k]
            )
        ric_action = self._simplex_project(ric_action)

        rho_k = np.array([
            min(1.0, entropies[k] / self.belief_engines[k].H_max +
                self.belief_engines[k].get_high_load_probability())
            for k in range(self.K)
        ])
        gamma = max(0, 1.0 - np.max(rho_k))

        a_guard_local = self._get_local_safe_action()
        final_action = self._simplex_project(
            gamma * ric_action + (1 - gamma) * a_guard_local
        )

        return self._physics_and_observation(final_action, raw_action, current_lam, gamma, entropies)


# ============================================================
# Ablation 2: w/o L4 SCU circuit-breaker
# ============================================================
class ATC_NoSCU(E2_Node_Simulator):
    """ATC 但跳过 L4 Safety Calibration Unit (security_unit.verify_and_fuse).

    Raw xApp action 不经过 L4 的 belief-triggered circuit-breaker, 直接
    送到 trust-aware fusion (L5/L6). 用来看 SCU 是不是必须的, 还是 L5+L6
    单独就够 safety.

    预期: violation rate 会 INCREASE (没有 L4 hard gate).
    """
    def step(self, xapp_action):
        if np.isscalar(xapp_action) or len(xapp_action) == 1:
            raw_action = np.full(self.K, float(xapp_action[0]) if not np.isscalar(xapp_action) else float(xapp_action))
        else:
            raw_action = np.array(xapp_action[:self.K], dtype=np.float64)

        # Adaptive telemetry runs normally
        for k in range(self.K):
            self.telemetry_controllers[k].adapt(self.belief_engines[k], tau=self.tau_per[k])

        entropies = np.array([be.get_entropy() for be in self.belief_engines])

        # 关键: 跳过 L4 SCU, raw_action 直接当作 ric_action 用
        ric_action = self._simplex_project(raw_action)
        current_lam = np.zeros(self.K)  # No lambda since no SCU

        rho_k = np.array([
            min(1.0, entropies[k] / self.belief_engines[k].H_max +
                self.belief_engines[k].get_high_load_probability())
            for k in range(self.K)
        ])
        gamma = max(0, 1.0 - np.max(rho_k))

        a_guard_local = self._get_local_safe_action()
        final_action = self._simplex_project(
            gamma * ric_action + (1 - gamma) * a_guard_local
        )

        return self._physics_and_observation(final_action, raw_action, current_lam, gamma, entropies)


# ============================================================
# Ablation 3: w/o L6 proportional reflex (use bang-bang)
# ============================================================
class ATC_BangBang(E2_Node_Simulator):
    """ATC 但 L6 guardrail 改成 binary bang-bang (而不是 proportional).

    把 _get_local_safe_action() 里的 proportional safe action 换成
    hard threshold: utilization > tau 就最大 throttle (0.95);
    否则 full release (0.2). 中间没有渐变带.

    预期: violation depth psi 会 INCREASE (突变 switching 容易 overshoot),
    spectral efficiency 也可能退化.
    """
    def _get_local_safe_action(self):
        """Bang-bang 版本: binary hard threshold, 没有渐变带."""
        a_guard = np.zeros(self.K)
        for k in range(self.K):
            if self.true_util[k] > self.tau_k[k]:
                a_guard[k] = 0.95  # Hard throttle
            else:
                a_guard[k] = 0.2   # Full release
        return a_guard


# ============================================================
# Helper used by 3 subclasses (extracts physics + obs from base env)
# ============================================================
def _add_physics_helper():
    """如果 E2_Node_Simulator 没有 _physics_and_observation 就注入一个.

    把 "final_action 算好之后" 的逻辑抽出来, 这样上面的 ablation 子类
    可以直接复用, 不用重复写 physics/sampling/observation 代码.
    """
    if hasattr(E2_Node_Simulator, '_physics_and_observation'):
        return

    def _physics_and_observation(self, final_action, raw_action, current_lam, gamma, entropies):
        # B. Physics evolution
        for k in range(self.K):
            traffic_flux = np.random.normal(0, self.sigma_k[k])
            if self.K == 1:
                action_effect = (0.5 - final_action[k]) * 0.1
                drift = 0.02
            else:
                neutral_action = 1.0 / self.K
                action_effect = (neutral_action - final_action[k]) * 0.1 * self.K
                drift = 0.01
            self.true_util[k] = np.clip(
                0.95 * self.true_util[k] + action_effect + traffic_flux + drift, 0, 1.0
            )

        # C. E2 sampling + belief update
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

        # D. Reward (matches base env Eq.7)
        r_sla = 0.0; r_perf = 0.0
        for k in range(self.K):
            u_k = self.true_util[k]; tau_k = self.tau_k[k]
            if u_k > tau_k:
                r_sla += -self.beta * (u_k - tau_k)
            diff = u_k - self.target_prb_k[k]
            r_perf += -(diff ** 2) * (40.0 if diff > 0 else 4.0)
        r_cost = -2.0 * total_sig_cost if (sampled_any and self.mode != "oracle") else 0.0
        total_reward = r_perf + r_sla + r_cost

        # E. Info dict
        avg_util = np.mean(self.true_util)
        avg_belief = np.mean([be.get_mean() for be in self.belief_engines])
        avg_entropy = np.mean(entropies)
        avg_lam = np.mean(current_lam) if isinstance(current_lam, np.ndarray) else current_lam
        avg_violation = float(np.any(self.true_util > self.tau_k))
        per_slice_violations = [float(self.true_util[k] > self.tau_k[k]) for k in range(self.K)]

        info = {
            "true_util": avg_util,
            "true_util_per_slice": self.true_util.copy(),
            "belief_mean": avg_belief,
            "belief_mean_per_slice": np.array([be.get_mean() for be in self.belief_engines]),
            "entropy": avg_entropy,
            "lambda": avg_lam,
            "gamma": gamma,
            "period": np.mean([tc.period for tc in self.telemetry_controllers]),
            "sig_cost": total_sig_cost,
            "is_violation": avg_violation,
            "per_slice_violations": per_slice_violations,
        }
        self.history.append(info)

        # F. Observation (proposed-mode shape)
        belief_means_now = np.array([be.get_mean() for be in self.belief_engines])
        entropies_now = np.array([be.get_entropy() for be in self.belief_engines])
        obs_vec = np.concatenate([
            [np.mean(self.last_reported_kpm), np.mean(self.tau_k)],
            belief_means_now, entropies_now, self.tau_k
        ]).astype(np.float32)

        return obs_vec, total_reward, False, False, info

    E2_Node_Simulator._physics_and_observation = _physics_and_observation


_add_physics_helper()
