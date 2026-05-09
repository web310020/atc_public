# ATC: Asynchronous Trust-aware Control

首先感谢您阅读这篇论文 First of all, thank you for reading this paper

## 简介

ATC = O-RAN 网络切片在 stale telemetry 下的层级 controller:

- POMDP belief engine 在 telemetry gap 期间重建隐藏的 network state
- Trust-aware fusion layer 根据 belief 清晰度, 在 strategic DRL agent 和 ms 级 local guardrail 之间分配控制权

## 环境

    pip install -r requirements.txt

跑过的环境: Python 3.11+, PyTorch 2.x, Gymnasium 0.29.x. 所有实验 CPU 就够, 不需要 GPU.

## 主要实验

    # K=3, 50 seeds
    python run_k3_seedbump_workers1.py --workers 1
    
    # K=5 probe (30 seeds)
    python run_k5_probe_workers1.py --workers 1
    
    # Ablation
    python train_l3_ablations.py && python run_l3_ablations_eval.py

输出落在 `experiments/<run_name>/`.

绘图(基于 `experiments/` 数据):

    python -m fig.plot_empirical_trap        # Fig 2
    python -m fig.plot_perf_distributions    # Fig 3
    python -m fig.plot_k_combined            # Fig 4

输出到 `paper_draft/figures/`.

## Authors

Zhiqiang Shen, Jitae Shin (Sungkyunkwan University)

## License

MIT, 见 [LICENSE](LICENSE).
