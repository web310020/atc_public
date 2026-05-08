# ATC: Asynchronous Trust-aware Control

首先感谢您阅读这篇论文 First of all, thank you for reading this paper



## 简介

ATC 是面向 O-RAN 网络切片、在 asynchronous telemetry 场景下的层级 controller:

- POMDP belief engine 在 telemetry gap 期间重建隐藏的 network state;
- Trust-aware fusion layer 根据 belief 清晰度, 在 strategic DRL agent 和 ms 级 local guardrail 之间分配控制权.

主结果: 3-slice 50-seed 配置下, ATC 在 aggregate utilization 上 match zero-delay full-information reference, worst-slice violation 大约降低 5 倍.

## 目录结构

    atc_public/
    ├── core/                              # 算法 + 仿真 env
    │   ├── core_logic.py                  # belief engine + safety calibration
    │   ├── telemetry_env.py               # asynchronous-telemetry 的 O-RAN sim
    │   ├── telemetry_env_ablations_v3.py  # per-component ablation 子类
    │   ├── k3_env.py                      # 多 slice (K >= 2) wrapper
    │   ├── k3_dirichlet_ppo.py            # Dirichlet PPO + per-slice Lagrangian
    │   ├── habib_hrl_baseline.py          # HRL baseline
    │   ├── batch_train.py                 # 训练辅助
    │   └── run_*.py                       # sensitivity / latency runner
    ├── fig/                               # 绘图脚本
    └── run_*.py                           # 顶层实验 orchestrator

## 环境

    pip install -r requirements.txt

跑过的环境: Python 3.11+, PyTorch 2.x, Gymnasium 0.29.x. 所有实验 CPU 就够, 不需要 GPU.

## 复现 paper 结果

| 内容                   | 命令                                                                              | 大约耗时 (CPU) |
|------------------------|-----------------------------------------------------------------------------------|---------------|
| K=1 主结果             | `python run_final_evaluation.py`                                                  | ~4 h          |
| K=3 主结果 (50 seeds)  | `python run_k3_seedbump_workers8.py --workers 8`                                  | ~12 h         |
| K=5 probe (30 seeds)   | `python run_k5_probe_workers8.py --workers 2`                                     | ~6 h          |
| Ablation 表            | `python train_l3_ablations.py && python run_l3_ablations_eval.py`                 | ~8 h          |
| Sensitivity 扫描       | `python run_supplementary_experiments.py --workers 4`                             | ~6 h          |
| Latency CDF            | `python core/run_latency_cdf.py --output experiments/latency_cdf --workers 1`     | ~30 min       |

每个 run 的输出落在 `experiments/<run_name>/` 下.

绘图(基于 `experiments/` 里的 run 数据):

    python -m fig.plot_empirical_trap        # Fig 2
    python -m fig.plot_perf_distributions    # Fig 3
    python -m fig.plot_k_combined            # Fig 4 (K=3 + K=5)

每个脚本输出 PDF + PNG 到 `paper_draft/figures/`.

## Quickstart 冒烟测试 (~5 min)

    python run_k3_experiment.py --seeds 1 --steps 10000 --template C

跑一个单 seed 的 K=3 短训练, 用同质 Template C (3 个相同 slice). 装好之后能正常跑完, 会写一个 `eval.md`.

## Authors

Zhiqiang Shen, Jitae Shin (Sungkyunkwan University)

## License

MIT, 见 [LICENSE](LICENSE).
