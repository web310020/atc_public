# core/k3_dirichlet_ppo.py
# ============================================================
# K=3 Dirichlet-Policy PPO with Per-Slice Lagrangian
# ============================================================
# Applies TOP-3 fixes from Belief_K3_algorithm_recommendations_20260424.md:
#   Fix 1: Dirichlet policy head (replaces softmax / Gaussian-Box)
#   Fix 3a: Per-slice reward normalization (RunningMeanStd per slice)
#   Fix 3b: Per-slice Lagrangian (K dual variables, one per SLA)
# Warm-start (Fix 2) is optional; see `load_warmstart_checkpoint`.
# ============================================================

import csv
import os
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.optim import Adam


# ============================================================
# Running statistics for observation / reward normalization
# ============================================================

class RunningMeanStd:
    """Online running mean/var (Welford-like). Serializable for checkpoints."""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1 and len(self.mean.shape) == 0:
            batch_mean, batch_var, batch_n = x.mean(), x.var(), x.size
        else:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_n = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_n
        new_mean = self.mean + delta * batch_n / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_n
        new_var = (m_a + m_b + np.square(delta) * self.count * batch_n / tot) / tot
        self.mean, self.var, self.count = new_mean, new_var, tot

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, d):
        self.mean = np.asarray(d["mean"])
        self.var = np.asarray(d["var"])
        self.count = d["count"]


# ============================================================
# Dirichlet Actor + Value Critic
# ============================================================

class DirichletActor(nn.Module):
    """
    Actor network: MLP encoder -> Dirichlet alpha head.
    Uses softplus + alpha_floor to keep alpha >= alpha_floor (default 1.0),
    which starts from a *uniform* Dirichlet distribution (safe exploration).
    This replaces the SB3 default DiagGaussian over Box[0,1]^K which causes
    the softmax-commit pathology on the K-simplex.
    """
    def __init__(self, obs_dim: int, K: int, hidden_dims=(256, 256),
                 alpha_floor: float = 1.0):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.alpha_head = nn.Linear(prev, K)
        # Orthogonal init with small gain on the output head prevents early saturation.
        nn.init.orthogonal_(self.alpha_head.weight, gain=0.01)
        nn.init.zeros_(self.alpha_head.bias)
        self.alpha_floor = alpha_floor
        self.K = K

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        raw = self.alpha_head(h)
        alpha = F.softplus(raw) + self.alpha_floor  # alpha >= alpha_floor
        return alpha

    def distribution(self, obs: torch.Tensor) -> Dirichlet:
        # validate_args=False: avoids torch's strict Simplex check rejecting
        # FP32-rounded samples whose sum differs from 1.0 by ~1e-7.
        # We clamp+renormalize actions ourselves before log_prob.
        return Dirichlet(self.forward(obs), validate_args=False)

    @staticmethod
    def _to_simplex(x: torch.Tensor) -> torch.Tensor:
        """Clean a near-simplex tensor: positive + exact sum=1 (FP-safe)."""
        x = x.clamp(min=1e-6)
        return x / x.sum(dim=-1, keepdim=True)

    def sample_action(self, obs: torch.Tensor, deterministic=False):
        dist = self.distribution(obs)
        if deterministic:
            alpha = dist.concentration
            # Mode of Dirichlet: (alpha-1)/(sum-K) when all alpha>1; else fallback to mean
            if (alpha > 1).all():
                mode = (alpha - 1) / (alpha.sum(dim=-1, keepdim=True) - alpha.shape[-1])
            else:
                mode = alpha / alpha.sum(dim=-1, keepdim=True)
            mode = self._to_simplex(mode)
            return mode, dist.log_prob(mode), dist.entropy()
        else:
            action = dist.rsample()
            action = self._to_simplex(action)
            return action, dist.log_prob(action), dist.entropy()

    def log_prob_entropy(self, obs, actions):
        dist = self.distribution(obs)
        # Clamp actions to open simplex to avoid log(0) in Dirichlet.log_prob
        actions = self._to_simplex(actions)
        return dist.log_prob(actions), dist.entropy()


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs)).squeeze(-1)


# ============================================================
# Per-slice Lagrangian  (Fix 3)
# ============================================================

class PerSliceLagrangian:
    """
    K independent dual variables lambda_k, one per slice's SLA budget.
    Update:  lambda_k <- max(0, lambda_k + lr * (E[viol_k] - budget_k))
    Per-slice reward normalization is tracked separately (see RunningMeanStd).
    """
    def __init__(self, K: int, sla_budget: np.ndarray, lr_dual: float = 1e-3,
                 lam_init: float = 0.0, lam_max: float = 50.0):
        self.K = K
        self.sla_budget = np.asarray(sla_budget, dtype=np.float64)  # per-slice violation rate caps
        self.lr_dual = lr_dual
        self.lam = np.full(K, lam_init, dtype=np.float64)
        self.lam_max = lam_max
        self.history = []

    def update(self, per_slice_violation_rate: np.ndarray):
        excess = per_slice_violation_rate - self.sla_budget
        self.lam = np.clip(self.lam + self.lr_dual * excess, 0.0, self.lam_max)
        self.history.append({"lam": self.lam.tolist(),
                             "viol": per_slice_violation_rate.tolist(),
                             "excess": excess.tolist()})

    def state_dict(self):
        return {"lam": self.lam.tolist(),
                "sla_budget": self.sla_budget.tolist(),
                "lr_dual": self.lr_dual,
                "lam_max": self.lam_max}

    def load_state_dict(self, d):
        self.lam = np.asarray(d["lam"])
        self.sla_budget = np.asarray(d["sla_budget"])
        self.lr_dual = d["lr_dual"]
        self.lam_max = d["lam_max"]


class GlobalSafeSliceLagrangian:
    """SafeSlice-style single global Lagrangian (CPO descendant per [SafeSlice_2025]).

    Unlike PerSliceLagrangian (K independent duals), maintains a single scalar lambda
    that ascends on aggregate (mean across slices) violation rate. This is a faithful
    reproduction of the [SafeSlice_2025] grid-searched Lagrangian formulation: the
    constraint signal is observation-conditioned (mean over per-slice violations), so
    under stale telemetry the dual update lacks the per-slice resolution of ATC's mu_k
    ascent and the policy collapses into the conservatism trap predicted in section II.B.
    """
    def __init__(self, sla_budget_global: float = 0.05, lr_dual: float = 1e-3,
                 lam_init: float = 0.0, lam_max: float = 100.0):
        self.sla_budget_global = float(sla_budget_global)
        self.lr_dual = float(lr_dual)
        self.lam = float(lam_init)
        self.lam_max = float(lam_max)
        self.history = []

    def update(self, per_slice_violation_rate):
        mean_viol = float(np.mean(per_slice_violation_rate))
        excess = mean_viol - self.sla_budget_global
        self.lam = float(np.clip(self.lam + self.lr_dual * excess, 0.0, self.lam_max))
        self.history.append({"lam": self.lam, "mean_viol": mean_viol,
                             "excess": float(excess)})

    def state_dict(self):
        return {"lam": self.lam, "sla_budget_global": self.sla_budget_global,
                "lr_dual": self.lr_dual, "lam_max": self.lam_max}

    def load_state_dict(self, d):
        self.lam = float(d["lam"])
        self.sla_budget_global = float(d["sla_budget_global"])
        self.lr_dual = float(d["lr_dual"])
        self.lam_max = float(d["lam_max"])


# ============================================================
# Rollout buffer
# ============================================================

@dataclass
class Rollout:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)          # shaped aggregate
    per_slice_rewards: List[np.ndarray] = field(default_factory=list)
    per_slice_violations: List[np.ndarray] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    true_util: List[np.ndarray] = field(default_factory=list)
    gamma_trust: List[float] = field(default_factory=list)

    def __len__(self):
        return len(self.obs)

    def to_arrays(self):
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "per_slice_rewards": np.asarray(self.per_slice_rewards, dtype=np.float32),
            "per_slice_violations": np.asarray(self.per_slice_violations, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "true_util": np.asarray(self.true_util, dtype=np.float32),
            "gamma_trust": np.asarray(self.gamma_trust, dtype=np.float32),
        }


# ============================================================
# PPO Trainer with Dirichlet + per-slice Lagrangian
# ============================================================

class K3DirichletPPO:
    def __init__(
        self,
        env_fn,                   # callable: seed -> env
        K: int = 3,
        sla_budget=(0.01, 0.05, 0.10),   # URLLC tight, eMBB mid, mMTC loose
        episode_len: int = 200,
        rollout_episodes: int = 8,       # 8 episodes per PPO update
        total_steps: int = 60_000,
        gamma_rl: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        ent_coef_final: float = 0.001,
        vf_coef: float = 0.5,
        lr_policy: float = 3e-4,
        lr_value: float = 3e-4,
        lr_dual: float = 1e-3,
        n_epochs: int = 8,
        batch_size: int = 256,
        hidden_dims=(256, 256),
        alpha_floor: float = 1.0,
        device: str = "cpu",
        normalize_obs: bool = True,
        normalize_per_slice_rewards: bool = True,
        use_lagrangian: bool = True,
        safeslice_mode: bool = False,
        seed: int = 0,
        verbose: bool = True,
        log_dir: Optional[str] = None,            # if set, stream per-update logs to disk
        checkpoint_every_updates: int = 10,        # 0 to disable periodic ckpt
        tag: str = "",                             # prefix for log files ("<mode>_seed<seed>")
    ):
        self.cfg = locals().copy()
        self.cfg.pop("self"); self.cfg.pop("env_fn")
        self.K = K
        self.seed = seed
        self.tag = tag or f"seed{seed}"
        torch.manual_seed(seed); np.random.seed(seed)

        # Streaming log setup
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._train_jsonl = (self.log_dir / f"{self.tag}_training_log.jsonl").open("a", buffering=1)
        else:
            self._train_jsonl = None
        self.checkpoint_every_updates = checkpoint_every_updates

        self.env = env_fn(seed)
        self.eval_env = env_fn(seed + 10_000)
        obs0, _ = self.env.reset(seed=seed)
        self.obs_dim = obs0.shape[0]

        self.device = torch.device(device)
        self.actor = DirichletActor(self.obs_dim, K, hidden_dims, alpha_floor).to(self.device)
        self.critic = ValueCritic(self.obs_dim, hidden_dims).to(self.device)
        self.opt_actor = Adam(self.actor.parameters(), lr=lr_policy)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr_value)

        self.episode_len = episode_len
        self.rollout_steps = episode_len * rollout_episodes
        self.total_steps = total_steps
        self.gamma_rl = gamma_rl
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef0 = ent_coef
        self.ent_coef_final = ent_coef_final
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_lagrangian = use_lagrangian
        self.safeslice_mode = safeslice_mode

        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,)) if normalize_obs else None
        self.per_slice_r_rms = [RunningMeanStd() for _ in range(K)] if normalize_per_slice_rewards else None
        if safeslice_mode:
            # SafeSlice: single global Lagrangian on aggregate violation (CPO descendant).
            # Mutually exclusive with per-slice Lagrangian (which is ATC's distinguishing mechanism).
            sla_budget_global = float(np.mean(np.asarray(sla_budget, dtype=np.float64)))
            self.lagrangian = None
            self.use_lagrangian = False
            self.safeslice_lagrangian = GlobalSafeSliceLagrangian(
                sla_budget_global=sla_budget_global, lr_dual=lr_dual)
        else:
            self.lagrangian = PerSliceLagrangian(K=K, sla_budget=np.asarray(sla_budget),
                                                 lr_dual=lr_dual) if use_lagrangian else None
            self.safeslice_lagrangian = None

        self.global_step = 0
        self.update_idx = 0
        self.train_log = []

    # ---------- rollout ----------
    def _normalize_obs(self, obs):
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs).astype(np.float32)

    def _shape_reward(self, raw_reward: float, per_slice_reward: np.ndarray,
                      per_slice_violation: np.ndarray) -> float:
        """Apply reward normalization and per-slice Lagrangian penalty."""
        if self.per_slice_r_rms is not None:
            normed = np.zeros(self.K)
            for k in range(self.K):
                # Update running stats with current batch point (single-step online)
                self.per_slice_r_rms[k].update(np.asarray([per_slice_reward[k]]))
                rk = per_slice_reward[k]
                sigma_k = math.sqrt(self.per_slice_r_rms[k].var) + 1e-8
                normed[k] = (rk - self.per_slice_r_rms[k].mean) / sigma_k
            base = normed.sum()
        else:
            base = float(raw_reward)
        if self.lagrangian is not None:
            base = base - float((self.lagrangian.lam * per_slice_violation).sum())
        elif self.safeslice_lagrangian is not None:
            # SafeSlice: subtract single global lambda * mean(per-slice violation)
            base = base - float(self.safeslice_lagrangian.lam * np.mean(per_slice_violation))
        return float(base)

    def collect_rollout(self) -> Rollout:
        roll = Rollout()
        obs, _ = self.env.reset(seed=self.seed + self.global_step)
        episode_counter = 0

        for t in range(self.rollout_steps):
            # Update obs stats online, normalize
            if self.obs_rms is not None:
                self.obs_rms.update(obs.reshape(1, -1))
            obs_n = self._normalize_obs(obs)
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, logp, _ = self.actor.sample_action(obs_t, deterministic=False)
                value = self.critic(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, r_raw, done, trunc, info = self.env.step(action_np)
            per_slice_r = np.asarray(info.get("per_slice_reward", np.zeros(self.K)), dtype=np.float64)
            per_slice_v = np.asarray(info.get("per_slice_violations", np.zeros(self.K)), dtype=np.float64)

            r_shaped = self._shape_reward(r_raw, per_slice_r, per_slice_v)

            roll.obs.append(obs_n)
            roll.actions.append(action_np)
            roll.log_probs.append(float(logp.item()))
            roll.values.append(float(value.item()))
            roll.rewards.append(r_shaped)
            roll.per_slice_rewards.append(per_slice_r)
            roll.per_slice_violations.append(per_slice_v)
            roll.true_util.append(np.asarray(info.get("true_util_per_slice", np.zeros(self.K))))
            roll.gamma_trust.append(float(info.get("gamma", 1.0)))

            episode_counter += 1
            ep_done = done or trunc or (episode_counter >= self.episode_len)
            roll.dones.append(bool(ep_done))
            if ep_done:
                obs, _ = self.env.reset(seed=self.seed + self.global_step + t)
                episode_counter = 0
            else:
                obs = next_obs

            self.global_step += 1

        # Bootstrap last value
        obs_n = self._normalize_obs(obs)
        with torch.no_grad():
            last_v = float(self.critic(torch.as_tensor(obs_n, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)).item())
        roll.values.append(last_v)   # trailing bootstrap value
        return roll

    # ---------- GAE ----------
    def compute_gae(self, rewards, values, dones):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]
            delta = rewards[t] + self.gamma_rl * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma_rl * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae
        returns = advantages + np.asarray(values[:-1], dtype=np.float32)
        return advantages, returns

    # ---------- PPO update ----------
    def _entropy_coef(self):
        # Linear anneal from ent_coef0 -> ent_coef_final across total_steps
        frac = min(1.0, self.global_step / max(1, self.total_steps))
        return self.ent_coef0 * (1 - frac) + self.ent_coef_final * frac

    def update(self, rollout: Rollout):
        arr = rollout.to_arrays()
        adv, ret = self.compute_gae(arr["rewards"], arr["values"], arr["dones"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(arr["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(arr["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(arr["log_probs"], dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        T = obs.shape[0]
        idx_all = np.arange(T)
        ent_coef = self._entropy_coef()
        losses = {"policy": [], "value": [], "entropy": [], "kl": []}

        for epoch in range(self.n_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, T, self.batch_size):
                mb = idx_all[start:start + self.batch_size]
                mb = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                new_logp, entropy = self.actor.log_prob_entropy(obs[mb], actions[mb])
                ratio = torch.exp(new_logp - old_logp[mb])
                pg1 = ratio * adv_t[mb]
                pg2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_t[mb]
                policy_loss = -torch.min(pg1, pg2).mean()

                value_pred = self.critic(obs[mb])
                value_loss = F.mse_loss(value_pred, ret_t[mb])

                ent = entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss - ent_coef * ent

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt_actor.step()
                self.opt_critic.step()

                with torch.no_grad():
                    kl = (old_logp[mb] - new_logp).mean().item()
                losses["policy"].append(policy_loss.item())
                losses["value"].append(value_loss.item())
                losses["entropy"].append(ent.item())
                losses["kl"].append(kl)

        # Dual ascent on lambda_k using rollout-averaged violation rate
        if self.lagrangian is not None:
            mean_viol = arr["per_slice_violations"].mean(axis=0)
            self.lagrangian.update(mean_viol)
        elif self.safeslice_lagrangian is not None:
            mean_viol = arr["per_slice_violations"].mean(axis=0)
            self.safeslice_lagrangian.update(mean_viol)

        self.update_idx += 1
        log = {
            "update": self.update_idx,
            "global_step": self.global_step,
            "wall_time_sec": round(time.time() - getattr(self, "_t_start", time.time()), 1),
            "policy_loss": float(np.mean(losses["policy"])),
            "value_loss": float(np.mean(losses["value"])),
            "entropy": float(np.mean(losses["entropy"])),
            "kl": float(np.mean(losses["kl"])),
            "ent_coef": ent_coef,
            "mean_return": float(arr["rewards"].mean()),
            "mean_util": float(arr["true_util"].mean()),
            "mean_gamma_trust": float(arr["gamma_trust"].mean()),
            "per_slice_viol_rate": arr["per_slice_violations"].mean(axis=0).tolist(),
            "per_slice_mean_util": arr["true_util"].mean(axis=0).tolist(),
            "per_slice_mean_reward": arr["per_slice_rewards"].mean(axis=0).tolist(),
            "lambda": (
                self.lagrangian.lam.tolist() if self.lagrangian
                else ([self.safeslice_lagrangian.lam] if self.safeslice_lagrangian
                      else [0.0] * self.K)
            ),
        }
        self.train_log.append(log)
        # Stream to disk so we don't lose anything if the run crashes.
        if self._train_jsonl is not None:
            self._train_jsonl.write(json.dumps(log) + "\n")
        # Periodic checkpoint so mid-run data is preserved.
        if (self.log_dir is not None and self.checkpoint_every_updates > 0
                and self.update_idx % self.checkpoint_every_updates == 0):
            ckpt = self.log_dir / f"{self.tag}_ckpt_upd{self.update_idx:04d}.pt"
            try:
                self.save(str(ckpt))
            except Exception as e:
                print(f"  (periodic ckpt failed: {e})")
        return log

    # ---------- training loop ----------
    def train(self):
        self._t_start = time.time()
        while self.global_step < self.total_steps:
            roll = self.collect_rollout()
            log = self.update(roll)
            if self.verbose and self.update_idx % 5 == 0:
                elapsed = time.time() - self._t_start
                print(f"  [{self.tag}] upd={log['update']:3d} step={self.global_step}/{self.total_steps} "
                      f"R={log['mean_return']:+.2f} U={log['mean_util']:.3f} "
                      f"viol={[f'{v:.3f}' for v in log['per_slice_viol_rate']]} "
                      f"lam={[f'{v:.2f}' for v in log['lambda']]} "
                      f"KL={log['kl']:+.4f} Ent={log['entropy']:.3f} ({elapsed:.0f}s)")
        # Close streaming log cleanly
        if self._train_jsonl is not None:
            self._train_jsonl.flush()
            self._train_jsonl.close()
            self._train_jsonl = None
        return self.train_log

    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True,
                 trajectory_csv_path: Optional[str] = None):
        """
        Evaluate the current policy. If trajectory_csv_path is set, dump
        one row per env step (for plotting + post-hoc analysis).
        """
        metrics = {
            "per_slice_viol_rate": np.zeros(self.K),
            "per_slice_mean_util": np.zeros(self.K),
            "aggregate_util": 0.0,
            "any_slice_violation_rate": 0.0,
            "mean_return": 0.0,
            "mean_gamma": 0.0,
            "total_sig_cost": 0.0,
        }
        n_steps = 0
        traj_rows = [] if trajectory_csv_path else None

        for ep in range(n_episodes):
            obs, _ = self.eval_env.reset(seed=self.seed + 99_000 + ep)
            ep_return = 0.0
            for t in range(self.episode_len):
                obs_n = self._normalize_obs(obs)
                obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, _, _ = self.actor.sample_action(obs_t, deterministic=deterministic)
                action_np = action.squeeze(0).cpu().numpy()
                obs, r, done, trunc, info = self.eval_env.step(action_np)
                ep_return += r
                util_k = np.asarray(info.get("true_util_per_slice", np.zeros(self.K)))
                viol_k = np.asarray(info.get("per_slice_violations", np.zeros(self.K)))
                metrics["per_slice_viol_rate"] += viol_k
                metrics["per_slice_mean_util"] += util_k
                metrics["aggregate_util"] += float(info.get("true_util", 0.0))
                metrics["any_slice_violation_rate"] += float(info.get("is_violation", 0.0))
                metrics["mean_gamma"] += float(info.get("gamma", 1.0))
                metrics["total_sig_cost"] += float(info.get("sig_cost", 0.0))
                n_steps += 1

                if traj_rows is not None:
                    row = {
                        "episode": ep, "step": t,
                        "reward": r, "gamma": float(info.get("gamma", 1.0)),
                        "sig_cost": float(info.get("sig_cost", 0.0)),
                        "aggregate_util": float(info.get("true_util", 0.0)),
                        "is_violation": float(info.get("is_violation", 0.0)),
                    }
                    for k in range(self.K):
                        row[f"action_k{k}"] = float(action_np[k])
                        row[f"util_k{k}"] = float(util_k[k])
                        row[f"viol_k{k}"] = float(viol_k[k])
                    traj_rows.append(row)

                if done or trunc:
                    break
            metrics["mean_return"] += ep_return

        metrics["per_slice_viol_rate"] = (metrics["per_slice_viol_rate"] / n_steps).tolist()
        metrics["per_slice_mean_util"] = (metrics["per_slice_mean_util"] / n_steps).tolist()
        metrics["aggregate_util"] = metrics["aggregate_util"] / n_steps
        metrics["any_slice_violation_rate"] = metrics["any_slice_violation_rate"] / n_steps
        metrics["mean_gamma"] = metrics["mean_gamma"] / n_steps
        metrics["mean_return"] = metrics["mean_return"] / n_episodes
        metrics["n_steps"] = n_steps
        metrics["n_episodes"] = n_episodes
        metrics["most_constrained_slice_viol"] = float(max(metrics["per_slice_viol_rate"]))

        if traj_rows is not None:
            csv_path = Path(trajectory_csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
                writer.writeheader(); writer.writerows(traj_rows)
            metrics["trajectory_csv"] = str(csv_path)

        return metrics

    # ---------- checkpointing ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms else None,
            "per_slice_r_rms": ([r.state_dict() for r in self.per_slice_r_rms]
                                if self.per_slice_r_rms else None),
            "lagrangian": self.lagrangian.state_dict() if self.lagrangian else None,
            "safeslice_lagrangian": (self.safeslice_lagrangian.state_dict()
                                     if self.safeslice_lagrangian else None),
            "cfg": {k: v for k, v in self.cfg.items()
                    if isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
            "global_step": self.global_step,
            "update_idx": self.update_idx,
        }
        torch.save(ckpt, path)

    def load(self, path: str, strict=True):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"], strict=strict)
        self.critic.load_state_dict(ckpt["critic"], strict=strict)
        if self.obs_rms and ckpt.get("obs_rms"):
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        if self.per_slice_r_rms and ckpt.get("per_slice_r_rms"):
            for r, d in zip(self.per_slice_r_rms, ckpt["per_slice_r_rms"]):
                r.load_state_dict(d)
        if self.lagrangian and ckpt.get("lagrangian"):
            self.lagrangian.load_state_dict(ckpt["lagrangian"])
        if self.safeslice_lagrangian and ckpt.get("safeslice_lagrangian"):
            self.safeslice_lagrangian.load_state_dict(ckpt["safeslice_lagrangian"])
        self.global_step = ckpt.get("global_step", 0)
        self.update_idx = ckpt.get("update_idx", 0)


# ============================================================
# Optional Fix 2 helper: grow policy head from K_old to K_new
# ============================================================

def grow_dirichlet_head(old_actor: DirichletActor, new_K: int,
                        new_alpha_floor: float = None) -> DirichletActor:
    """
    For Fix 2 (curriculum K-ramp). Take an actor trained on K_old slices
    and return a new actor with K_new slices; encoder weights are copied,
    old K_old slice columns in the head are copied, new slice columns
    are initialized near zero (gives ~alpha_floor uniform start on new dim).
    """
    old_K = old_actor.K
    assert new_K >= old_K, "grow only (new_K >= old_K)"
    hidden = tuple(l.out_features for l in old_actor.encoder if isinstance(l, nn.Linear))
    obs_dim = [l for l in old_actor.encoder if isinstance(l, nn.Linear)][0].in_features
    floor = new_alpha_floor if new_alpha_floor is not None else old_actor.alpha_floor
    new_actor = DirichletActor(obs_dim, new_K, hidden, alpha_floor=floor)
    # copy encoder
    new_actor.encoder.load_state_dict(old_actor.encoder.state_dict())
    # copy old K_old output rows; new rows start near zero
    with torch.no_grad():
        new_actor.alpha_head.weight[:old_K, :] = old_actor.alpha_head.weight.clone()
        new_actor.alpha_head.bias[:old_K] = old_actor.alpha_head.bias.clone()
        # leave new rows at their orthogonal_(0.01) init
    return new_actor
