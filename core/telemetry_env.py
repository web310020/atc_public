# telemetry_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from core.core_logic import (
    Belief_Manager_KF, Belief_Manager,
    Adaptive_Telemetry_Controller, Security_Control_Unit
)


# ============================================================
# Slice Traffic Profiles (heterogeneous per-slice characteristics)
# ============================================================

SLICE_PROFILES = {
    "eMBB":  {"sigma": 0.05, "base_util": 0.40, "target_prb": 0.70, "label": "eMBB"},
    "V2X":   {"sigma": 0.03, "base_util": 0.35, "target_prb": 0.65, "label": "V2X"},
    "URLLC": {"sigma": 0.08, "base_util": 0.30, "target_prb": 0.60, "label": "URLLC"},
}

DEFAULT_SLICE_ORDER = ["eMBB", "V2X", "URLLC"]


class E2_Node_Simulator(gym.Env):
    """
    O-RAN E2 Node Simulator with K-slice support.
    Each slice has independent traffic dynamics, belief engine, and SLA target.
    Actions are K-dimensional (simplex-constrained PRB allocation).
    """

    def __init__(self, mode="proposed", K=3, tau=0.5, base_period=200,
                 beta=200, use_kalman=True, slice_profiles=None):
        super(E2_Node_Simulator, self).__init__()
        self.mode = mode
        self.K = K
        self.base_period = base_period
        self.beta = beta
        self.use_kalman = use_kalman

        # Per-slice configuration
        if slice_profiles is None:
            if K <= len(DEFAULT_SLICE_ORDER):
                self.slice_names = DEFAULT_SLICE_ORDER[:K]
            else:
                self.slice_names = DEFAULT_SLICE_ORDER + [f"Slice_{i}" for i in range(3, K)]
        else:
            self.slice_names = list(slice_profiles.keys())[:K]

        self.sigma_k = np.array([
            SLICE_PROFILES.get(name, {"sigma": 0.05})["sigma"]
            for name in self.slice_names
        ])
        self.target_prb_k = np.array([
            SLICE_PROFILES.get(name, {"target_prb": 0.70})["target_prb"]
            for name in self.slice_names
        ])
        self.base_util_k = np.array([
            SLICE_PROFILES.get(name, {"base_util": 0.40})["base_util"]
            for name in self.slice_names
        ])

        # Per-slice SLA thresholds (can be heterogeneous)
        self.tau_k = np.full(K, tau)
        self.target_latency_sla = tau  # Global SLA (for backward compat)

        # Observation: [global_kpm_avg, global_sla] + [belief_mean_1..K] + [entropy_1..K] + [tau_1..K]
        obs_dim = 2 + 3 * K
        self.observation_space = spaces.Box(low=0, high=2, shape=(obs_dim,), dtype=np.float32)

        # Action: K-dimensional normalized PRB allocation
        self.action_space = spaces.Box(low=0, high=1, shape=(K,), dtype=np.float32)

        # Per-slice components
        if use_kalman:
            self.belief_engines = [Belief_Manager_KF(obs_noise_std=self.sigma_k[k]) for k in range(K)]
        else:
            self.belief_engines = [Belief_Manager() for _ in range(K)]

        self.telemetry_controllers = [Adaptive_Telemetry_Controller() for _ in range(K)]
        self.security_unit = Security_Control_Unit()

        # Per-slice state
        self.true_util = np.copy(self.base_util_k)
        self.last_reported_kpm = np.copy(self.base_util_k)
        self.steps_since_last_indication = np.zeros(K, dtype=int)
        self.sim_step_ms = 10

        # Adaptive telemetry sensitivity (per-slice)
        self.tau_per = np.full(K, 0.9)  # perception threshold

        self.history = []

    def _simplex_project(self, action):
        """Project action onto simplex: sum(a) <= 1, a >= 0."""
        action = np.clip(action, 0, 1)
        total = np.sum(action)
        if total > 1.0:
            action = action / total
        return action

    def step(self, xapp_action):
        # Ensure action is K-dimensional
        if np.isscalar(xapp_action) or len(xapp_action) == 1:
            raw_action = np.full(self.K, float(xapp_action[0]) if not np.isscalar(xapp_action) else float(xapp_action))
        else:
            raw_action = np.array(xapp_action[:self.K], dtype=np.float64)

        # Per-slice entropy and belief state
        entropies = np.array([be.get_entropy() for be in self.belief_engines])
        belief_means = np.array([be.get_mean() for be in self.belief_engines])

        # --- A. Mode-specific control logic ---
        if self.mode == "safeslice":
            final_action = self._simplex_project(raw_action)
            current_lam = np.ones(self.K)
            gamma = 1.0
            for tc in self.telemetry_controllers:
                tc.period = self.base_period

        elif self.mode == "proposed":
            # 1. Per-slice adaptive telemetry
            for k in range(self.K):
                self.telemetry_controllers[k].adapt(self.belief_engines[k], tau=self.tau_per[k])

            # 2. Per-slice RIC-side safety fusion
            ric_action = np.zeros(self.K)
            current_lam = np.zeros(self.K)
            for k in range(self.K):
                ric_action[k], current_lam[k] = self.security_unit.verify_and_fuse(
                    raw_action[k], self.belief_engines[k]
                )
            ric_action = self._simplex_project(ric_action)

            # 3. Global Trust Index: worst-case scalarization (Eq.12 in paper)
            rho_k = np.array([
                min(1.0, entropies[k] / self.belief_engines[k].H_max +
                    self.belief_engines[k].get_high_load_probability())
                for k in range(self.K)
            ])
            gamma = max(0, 1.0 - np.max(rho_k))

            # 4. Per-slice BS-side local protection
            a_guard_local = self._get_local_safe_action()

            # 5. Trust-aware fusion
            final_action = self._simplex_project(
                gamma * ric_action + (1 - gamma) * a_guard_local
            )

        elif self.mode == "static_slicing":
            final_action = np.full(self.K, 1.0 / self.K)
            current_lam = np.zeros(self.K)
            gamma = 1.0
            for tc in self.telemetry_controllers:
                tc.period = self.base_period

        elif self.mode == "guardrail_only":
            ric_action = np.zeros(self.K)
            current_lam = np.zeros(self.K)
            for k in range(self.K):
                ric_action[k], current_lam[k] = self.security_unit.verify_and_fuse(
                    0.5, self.belief_engines[k]
                )
            final_action = self._simplex_project(ric_action)
            gamma = 1.0
            for tc in self.telemetry_controllers:
                tc.period = self.base_period

        elif self.mode == "oracle":
            final_action = self._simplex_project(raw_action)
            current_lam = np.zeros(self.K)
            gamma = 1.0
            for tc in self.telemetry_controllers:
                tc.period = 10

        else:  # vanilla_ppo, lstm_predictive, etc.
            final_action = self._simplex_project(raw_action)
            current_lam = np.zeros(self.K)
            gamma = 1.0
            for tc in self.telemetry_controllers:
                tc.period = self.base_period

        # --- B. Per-slice physics evolution ---
        for k in range(self.K):
            traffic_flux = np.random.normal(0, self.sigma_k[k])
            if self.K == 1:
                # K=1: pivot 在 0.5, baseline drift +0.02
                action_effect = (0.5 - final_action[k]) * 0.1
                drift = 0.02
            else:
                # Multi-slice: pivot at 1/K, drift scaled
                neutral_action = 1.0 / self.K
                action_effect = (neutral_action - final_action[k]) * 0.1 * self.K
                drift = 0.01
            self.true_util[k] = np.clip(
                0.95 * self.true_util[k] + action_effect + traffic_flux + drift,
                0, 1.0
            )

        # --- C. Per-slice E2 sampling and belief update ---
        total_sig_cost = 0.0
        sampled_any = False
        for k in range(self.K):
            current_period = self.telemetry_controllers[k].period
            self.steps_since_last_indication[k] += self.sim_step_ms

            if self.steps_since_last_indication[k] >= current_period:
                self.last_reported_kpm[k] = self.true_util[k]
                self.belief_engines[k].update(self.last_reported_kpm[k])
                self.steps_since_last_indication[k] = 0
                total_sig_cost += 1.0
                sampled_any = True
            else:
                self.belief_engines[k].predict(final_action[k], K=self.K)

        # --- D. Aggregate reward (Eq.7 in paper: sum over K slices) ---
        r_sla = 0.0
        r_perf = 0.0
        for k in range(self.K):
            u_k = self.true_util[k]
            tau_k = self.tau_k[k]
            # SLA violation penalty
            if u_k > tau_k:
                r_sla += -self.beta * (u_k - tau_k)
            # Performance: penalize deviation from target
            diff = u_k - self.target_prb_k[k]
            if diff > 0:
                r_perf += -(diff ** 2) * 40.0
            else:
                r_perf += -(diff ** 2) * 4.0

        r_cost = -2.0 * total_sig_cost if (sampled_any and self.mode != "oracle") else 0.0
        total_reward = r_perf + r_sla + r_cost

        # --- E. Record metrics ---
        avg_util = np.mean(self.true_util)
        avg_belief = np.mean([be.get_mean() for be in self.belief_engines])
        avg_entropy = np.mean(entropies)
        avg_lam = np.mean(current_lam) if isinstance(current_lam, np.ndarray) else current_lam
        avg_violation = float(np.any(self.true_util > self.tau_k))

        # Per-slice violation tracking
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

        # --- F. Observation vector: [avg_kpm, avg_sla, belief_mean_1..K, entropy_1..K, tau_1..K] ---
        belief_means_now = np.array([be.get_mean() for be in self.belief_engines])
        entropies_now = np.array([be.get_entropy() for be in self.belief_engines])

        if self.mode == "proposed":
            obs_vec = np.concatenate([
                [np.mean(self.last_reported_kpm), np.mean(self.tau_k)],
                belief_means_now,
                entropies_now,
                self.tau_k
            ]).astype(np.float32)
        elif self.mode == "safeslice":
            obs_vec = np.concatenate([
                [np.mean(self.last_reported_kpm), np.mean(self.tau_k)],
                np.zeros(self.K),
                np.full(self.K, 1.5),
                self.tau_k
            ]).astype(np.float32)
        else:
            obs_vec = np.concatenate([
                [np.mean(self.last_reported_kpm), np.mean(self.tau_k)],
                np.zeros(self.K),
                np.full(self.K, 1.5),
                self.tau_k
            ]).astype(np.float32)

        return obs_vec, total_reward, False, False, info

    def _get_local_safe_action(self):
        """Per-slice BS-side proportional safety reflex (1ms local loop).
        Gradual response as utilization approaches SLA boundary."""
        margin_width = 0.15  # Proportional activation zone
        a_guard = np.zeros(self.K)
        for k in range(self.K):
            margin = self.tau_k[k] - self.true_util[k]
            if margin < 0:
                a_guard[k] = 0.95       # Already violating: max throttle
            elif margin < margin_width:
                a_guard[k] = 0.2 + 0.75 * (1.0 - margin / margin_width)  # Gradual
            else:
                a_guard[k] = 0.2        # Safe zone: minimal intervention
        return self._simplex_project(a_guard)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.true_util = np.copy(self.base_util_k)
        self.last_reported_kpm = np.copy(self.base_util_k)
        self.steps_since_last_indication = np.zeros(self.K, dtype=int)
        self.history = []

        # Reset belief engines
        if self.use_kalman:
            self.belief_engines = [
                Belief_Manager_KF(obs_noise_std=self.sigma_k[k]) for k in range(self.K)
            ]
        else:
            self.belief_engines = [Belief_Manager() for _ in range(self.K)]
        self.telemetry_controllers = [Adaptive_Telemetry_Controller() for _ in range(self.K)]

        # Initial observation
        obs_vec = np.concatenate([
            [np.mean(self.base_util_k), np.mean(self.tau_k)],
            np.full(self.K, 0.5),  # initial belief means
            np.full(self.K, 0.5),  # initial entropies
            self.tau_k
        ]).astype(np.float32)

        return obs_vec, {}

    def set_tau(self, new_tau):
        """Adjust perception sensitivity (called by A1 interface)."""
        self.tau_per = np.full(self.K, new_tau)

    def set_a1_policy(self, new_sla):
        """A1 intent injection: update per-slice SLA thresholds."""
        self.target_latency_sla = new_sla
        self.tau_k = np.full(self.K, new_sla)
        # Adaptive perception threshold linked to SLA
        alpha_log = 1.0
        self.tau_per = np.full(self.K, alpha_log * np.log(1 + new_sla))

    def get_academic_stats(self):
        """Return DataFrame of episode history + summary stats."""
        if not self.history:
            return pd.DataFrame(), {}
        df = pd.DataFrame(self.history)
        stats = {
            "avg_util": df["true_util"].mean(),
            "violation_rate": df["is_violation"].mean(),
            "total_cost": df["sig_cost"].sum()
        }
        return df, stats
