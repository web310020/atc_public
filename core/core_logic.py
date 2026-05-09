# core/core_logic.py
# ============================================================
# Belief engine + adaptive telemetry + safety fusion 模块.
#
# 主要组件:
#   - Belief_Manager_KF: Kalman 连续 belief (主用, 配 ATC)
#   - Belief_Manager:    discrete 3-state POMDP belief (备用, K=1)
#   - Adaptive_Telemetry_Controller: entropy-driven E2 sampling 周期
#   - Security_Control_Unit: trust-aware fusion (AI action vs guardrail)
#   - LSTMPredictor / LSTMController: pure-LSTM 预测 baseline
# ============================================================

import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn


# ---------- Kalman 连续 belief ----------

class Belief_Manager_KF:
    """连续 Gaussian utilization tracker (替 discrete 3-state POMDP)."""

    def __init__(self, process_noise=0.005, obs_noise_std=0.05):
        self.mu = 0.5
        self.P = 0.1
        self.Q = process_noise
        self.R = obs_noise_std ** 2
        self.A = 0.95
        self.obs_noise_std = obs_noise_std

        # P_max ≈ 20-step gap × Q + initial; H_max 兼容 discrete 3-state 阈值
        self.P_max = 0.15
        self.H_max = 1.58

    def predict(self, action, K=1):
        if K == 1:
            action_effect = (0.5 - action) * 0.1
            drift = 0.02
        else:
            neutral = 1.0 / K
            action_effect = (neutral - action) * 0.1 * K
            drift = 0.01
        self.mu = np.clip(self.A * self.mu + action_effect + drift, 0, 1.0)
        self.P = self.A ** 2 * self.P + self.Q
        return self.mu

    def update(self, observation):
        K = self.P / (self.P + self.R)
        self.mu = np.clip(self.mu + K * (observation - self.mu), 0, 1.0)
        self.P = (1 - K) * self.P
        return self.mu

    def get_entropy(self):
        # P ∈ [0, P_max] → [0, H_max], 对齐原 discrete 阈值
        normalized = min(self.P / self.P_max, 1.0) * self.H_max
        return normalized

    def get_mean(self):
        return self.mu

    def get_variance(self):
        return self.P

    def get_high_load_probability(self, threshold=0.5):
        if self.P <= 0:
            return 1.0 if self.mu > threshold else 0.0
        return 1.0 - norm.cdf(threshold, loc=self.mu, scale=np.sqrt(max(self.P, 1e-12)))

    def reset(self):
        self.mu = 0.5
        self.P = 0.1


# ---------- discrete 3-state belief (legacy, K=1 时用) ----------

class Belief_Manager:
    """备用 discrete 3-state POMDP belief, K=1 时用."""

    def __init__(self, num_states=3):
        self.num_states = num_states
        self.belief = np.full(num_states, 1.0 / num_states)

        if num_states == 3:
            self.base_transition = np.array([
                [0.85, 0.15, 0.00],
                [0.10, 0.80, 0.10],
                [0.00, 0.15, 0.85]
            ])
        else:
            self.base_transition = self._generate_transition_matrix(num_states)

        self.state_centroids = np.linspace(0.15, 0.85, num_states)
        self.observation_models = {
            s: {'mu': self.state_centroids[s], 'sigma': 0.05}
            for s in range(num_states)
        }
        self.H_max = np.log2(num_states)

    def _generate_transition_matrix(self, n):
        P = np.zeros((n, n))
        for i in range(n):
            P[i, i] = 0.80
            if i > 0:
                P[i, i - 1] = 0.10
            if i < n - 1:
                P[i, i + 1] = 0.10
            P[i] /= P[i].sum()
        return P

    def predict(self, action, K=1):
        prior = self.base_transition.T @ self.belief
        alpha = 0.18
        shift = (0.5 - action) * alpha

        new_belief = np.zeros(self.num_states)
        if self.num_states == 3:
            if shift > 0:
                s = min(shift, 0.9)
                new_belief[2] = prior[2] + prior[1] * s
                new_belief[1] = prior[1] * (1 - s) + prior[0] * s
                new_belief[0] = prior[0] * (1 - s)
            else:
                s = min(abs(shift), 0.9)
                new_belief[0] = prior[0] + prior[1] * s
                new_belief[1] = prior[1] * (1 - s) + prior[2] * s
                new_belief[2] = prior[2] * (1 - s)
        else:
            s = min(abs(shift), 0.9)
            direction = 1 if shift > 0 else -1
            for i in range(self.num_states):
                new_belief[i] = prior[i] * (1 - s)
                neighbor = i + direction
                if 0 <= neighbor < self.num_states:
                    new_belief[i] += prior[neighbor] * s

        self.belief = new_belief / (np.sum(new_belief) + 1e-9)
        return self.belief

    def update(self, observation):
        prior = self.base_transition.T @ self.belief
        likelihoods = np.zeros(self.num_states)
        for s in range(self.num_states):
            mu = self.observation_models[s]['mu']
            sigma = self.observation_models[s]['sigma']
            likelihoods[s] = norm.pdf(observation, loc=mu, scale=sigma)
        unnormalized_posterior = likelihoods * prior
        evidence = np.sum(unnormalized_posterior)
        self.belief = unnormalized_posterior / (evidence + 1e-9)
        return self.belief

    def get_entropy(self):
        return -np.sum(self.belief * np.log2(self.belief + 1e-9))

    def get_mean(self):
        return np.dot(self.belief, self.state_centroids)

    def get_high_load_probability(self, threshold=None):
        return self.belief[-1]

    def reset(self):
        self.belief = np.full(self.num_states, 1.0 / self.num_states)


# ---------- Adaptive telemetry controller ----------

class Adaptive_Telemetry_Controller:
    def __init__(self):
        self.period = 200

    def adapt(self, belief_engine, tau=0.9):
        # entropy 高 / 高负载概率高 → 缩短 sampling 周期
        entropy = belief_engine.get_entropy()
        high_risk = belief_engine.get_high_load_probability()
        if entropy > tau or high_risk > 0.5:
            self.period = 10
        else:
            self.period = 200
        return self.period


# ---------- Security control unit (trust-aware fusion) ----------

class Security_Control_Unit:
    def __init__(self, base_lam=0.3):
        self.base_lam = base_lam

    def get_guardrail_action(self, belief_engine):
        # 基于 belief mean 的保守 fallback action
        mu = belief_engine.get_mean()
        if mu > 0.65:
            return 0.95
        elif mu < 0.25:
            return 0.05
        return 0.5

    def verify_and_fuse(self, ai_action, belief_engine):
        # AI action 与 guardrail 的 risk-weighted 融合
        mu = belief_engine.get_mean()
        entropy = belief_engine.get_entropy()
        H_max = belief_engine.H_max
        high_risk = belief_engine.get_high_load_probability()

        if mu > 0.60:
            risk_factor = 0.99
        else:
            risk_factor = np.clip(entropy / H_max + high_risk, self.base_lam, 0.9)

        guardrail_act = self.get_guardrail_action(belief_engine)
        final_action = (1 - risk_factor) * ai_action + risk_factor * guardrail_act
        return final_action, risk_factor


# ---------- LSTM baseline ----------

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMController:
    def __init__(self):
        self.model = LSTMPredictor()
        self.history = []
        self.seq_len = 10
        self.period = 200

    def predict_and_adapt(self, obs):
        self.history.append([obs])
        if len(self.history) > self.seq_len: self.history.pop(0)
        if len(self.history) < self.seq_len: return 200
        input_tensor = torch.FloatTensor([self.history])
        with torch.no_grad():
            prediction = self.model(input_tensor).item()
        self.period = 10 if prediction > 0.75 else 200
        return self.period
