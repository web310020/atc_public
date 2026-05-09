"""
ATC ablation subclasses (隔离单组件贡献).

每个子类只 flip ONE component, 其它保持 "proposed" mode 不变:
  - ATC_NoAdaptiveTelemetry: 固定 base_period, 关 entropy-gated 自适应
  - ATC_NoSCU:               跳过 RIC-side safety unit (raw action 直送融合)
  - ATC_BangBang:            BS-side reflex 改 binary 阈值 (无 proportional 区)
  - Kalman belief 变体: E2_Node_Simulator(use_kalman=True)
"""

import numpy as np
from core.telemetry_env import E2_Node_Simulator


# ---------- 关 entropy-driven 自适应 telemetry ----------
class ATC_NoAdaptiveTelemetry(E2_Node_Simulator):
    """固定 telemetry period (无 entropy-gated 自适应)."""
    def step(self, xapp_action):
        if np.isscalar(xapp_action) or len(xapp_action) == 1:
            raw_action = np.full(self.K, float(xapp_action[0]) if not np.isscalar(xapp_action) else float(xapp_action))
        else:
            raw_action = np.array(xapp_action[:self.K], dtype=np.float64)

        entropies = np.array([be.get_entropy() for be in self.belief_engines])

        # 强制固定 period
        for tc in self.telemetry_controllers:
            tc.period = self.base_period

        # 其余沿用 "proposed" 逻辑
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


# ---------- 关 RIC-side safety unit ----------
class ATC_NoSCU(E2_Node_Simulator):
    """跳过 security_unit.verify_and_fuse, raw action 直送融合."""
    def step(self, xapp_action):
        if np.isscalar(xapp_action) or len(xapp_action) == 1:
            raw_action = np.full(self.K, float(xapp_action[0]) if not np.isscalar(xapp_action) else float(xapp_action))
        else:
            raw_action = np.array(xapp_action[:self.K], dtype=np.float64)

        # 自适应 telemetry 正常跑
        for k in range(self.K):
            self.telemetry_controllers[k].adapt(self.belief_engines[k], tau=self.tau_per[k])

        entropies = np.array([be.get_entropy() for be in self.belief_engines])

        # 跳过 SCU, raw action 直送
        ric_action = self._simplex_project(raw_action)
        current_lam = np.zeros(self.K)

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


# ---------- BS-side reflex 改 binary (bang-bang) ----------
class ATC_BangBang(E2_Node_Simulator):
    """BS-side reflex: 二值阈值 (无 proportional gradient zone)."""
    def _get_local_safe_action(self):
        a_guard = np.zeros(self.K)
        for k in range(self.K):
            if self.true_util[k] > self.tau_k[k]:
                a_guard[k] = 0.95
            else:
                a_guard[k] = 0.2
        return a_guard


# ---------- helper: physics + obs 注入 base env (避免子类重复) ----------
def _add_physics_helper():
    if hasattr(E2_Node_Simulator, '_physics_and_observation'):
        return

    def _physics_and_observation(self, final_action, raw_action, current_lam, gamma, entropies):
        # physics 演进
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

        # E2 sampling + belief update
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

        # reward
        r_sla = 0.0; r_perf = 0.0
        for k in range(self.K):
            u_k = self.true_util[k]; tau_k = self.tau_k[k]
            if u_k > tau_k:
                r_sla += -self.beta * (u_k - tau_k)
            diff = u_k - self.target_prb_k[k]
            r_perf += -(diff ** 2) * (40.0 if diff > 0 else 4.0)
        r_cost = -2.0 * total_sig_cost if (sampled_any and self.mode != "oracle") else 0.0
        total_reward = r_perf + r_sla + r_cost

        # info dict
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

        # obs (proposed-mode shape)
        belief_means_now = np.array([be.get_mean() for be in self.belief_engines])
        entropies_now = np.array([be.get_entropy() for be in self.belief_engines])
        obs_vec = np.concatenate([
            [np.mean(self.last_reported_kpm), np.mean(self.tau_k)],
            belief_means_now, entropies_now, self.tau_k
        ]).astype(np.float32)

        return obs_vec, total_reward, False, False, info

    E2_Node_Simulator._physics_and_observation = _physics_and_observation


_add_physics_helper()
