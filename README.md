# BigSnakeRL — Snake Reinforcement Learning (中文 / English)

**Language / 语言**: **[中文](#中文说明)** | **[English](#english-guide)**

---

# 中文说明

## 项目简介
BigSnakeRL 是一个面向强化学习教学与研究的贪吃蛇项目，包含：
- Gymnasium 风格环境（`envs/big_snake_env.py`）
- Double DQN 训练脚本 + 课程学习（`train_dqn_snake.py`）
- Checkpoint 保存与断点续训
- 训练统计输出 + 得分趋势图（`score_trend.png`）
- pygame 可视化（渲染不吞事件）

---

## 目录结构
```text
BigSnakeRL/
├── envs/
│   └── big_snake_env.py
├── train_dqn_snake.py
├── checkpoints/            # 自动生成
│   ├── stage_1.pt
│   ├── stage_2.pt
│   └── stage_3.pt
├── score_trend.png         # 自动生成
└── README.md
```

---

## 运行环境与依赖

### 1) 创建并激活 conda 环境（示例）
```powershell
conda create -n bigsnake python=3.10
conda activate bigsnake
```

### 2) 安装依赖
基础依赖：
```powershell
pip install gymnasium pygame matplotlib numpy
```

GPU（CUDA 12.8 / cu128）版 PyTorch：
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 当前训练脚本为“强制 GPU”版本：如果 CUDA 不可用会直接报错退出。  
> 我们希望未来有人贡献 CPU-only 支持（见「贡献与路线图」）。

### 3) 运行训练
```powershell
python .\train_dqn_snake.py
```

---

## 课程学习（先学什么，再学什么）

训练通过 `STAGE` 分三阶段进行（修改 `train_dqn_snake.py` 顶部）：

- **Stage 1（入门）**：小地图 + 强正奖励 + 强距离 shaping，先学会“追果子并吃到”
- **Stage 2（进阶）**：中地图 + 轻度 stall/loop 惩罚，学会“稳定吃 + 不拖延”
- **Stage 3（最终）**：大地图 + 完整 stall/loop 惩罚，学会“更接近真实策略”

建议流程：
1. `STAGE=1` 训练到平均 score 明显上升
2. 改为 `STAGE=2` 继续训练
3. 最后 `STAGE=3` 完成最终训练

---

## Checkpoint（关键：不会每次从头开始）
- 训练过程中会自动保存：
  - `checkpoints/stage_1.pt`
  - `checkpoints/stage_2.pt`
  - `checkpoints/stage_3.pt`
- 下次运行同一个 `STAGE` 会自动加载并续训

---

## 输出与可视化
- 每隔 `SHOW_EVERY` 局：
  - 输出最近窗口的平均 score 与 epsilon
  - 输出 action 分布（验证是否只走一个方向）
  - 演示一局当前策略（pygame 渲染）
- 训练结束：
  - 生成 `score_trend.png` 并弹出图表窗口

---

## 已知问题（后续重点）
### 1) 蛇疯狂转圈（looping）
可能原因：
- reward 局部最优（例如靠近食物的 shaping 与回避惩罚冲突）
- 状态表示缺乏“历史/覆盖率”信息

候选解决方向：
- 更强的循环检测（轨迹重复惩罚、覆盖率奖励）
- curiosity / novelty bonus
- PPO / A2C 等更稳定算法尝试
- 引入更丰富的观测（方向、相对坐标、局部视野）

### 2) 蛇主动自杀（suicidal policy）
可能原因：
- 负奖励过强或 early training 信号稀疏
- 探索策略与终止奖励设计不平衡

候选解决方向：
- 生存奖励（time-alive bonus）
- 更合理的 reward 标度（reward scaling）
- 分布式/分布价值（distributional RL）

---

## 贡献与路线图（欢迎 PR）
我们非常希望有人加入：
- **CPU-only 支持**（自动 device 选择：cuda/cpu；并提供低算力默认网络）
- 更稳定的 anti-loop / anti-suicide 机制
- 新算法支持：PPO / Rainbow DQN / C51 / NoisyNet 等
- 工程化：TensorBoard、配置文件化（YAML）、多环境并行采样

---

# English Guide

## Overview
BigSnakeRL is a Snake environment designed for reinforcement learning (RL) education and experiments. It includes:
- A Gymnasium-style environment (`envs/big_snake_env.py`)
- A Double DQN training script with curriculum learning (`train_dqn_snake.py`)
- Automatic checkpointing and resume training
- Training metrics + score trend plot (`score_trend.png`)
- pygame rendering (render does not consume the event queue)

---

## Project Layout
```text
BigSnakeRL/
├── envs/
│   └── big_snake_env.py
├── train_dqn_snake.py
├── checkpoints/            # auto-generated
│   ├── stage_1.pt
│   ├── stage_2.pt
│   └── stage_3.pt
├── score_trend.png         # auto-generated
└── README.md
```

---

## Setup & Dependencies

### 1) Create and activate a conda environment (example)
```powershell
conda create -n bigsnake python=3.10
conda activate bigsnake
```

### 2) Install dependencies
Base:
```powershell
pip install gymnasium pygame matplotlib numpy
```

PyTorch (CUDA 12.8 / cu128):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> The training script currently **requires CUDA** and will exit if CUDA is unavailable.  
> We explicitly welcome contributions to add a CPU-only mode (see “Contributing & Roadmap”).

### 3) Run training
```powershell
python .\train_dqn_snake.py
```

---

## Curriculum Learning (What to learn first, then what)
Training is split into three stages controlled by `STAGE` at the top of `train_dqn_snake.py`:

- **Stage 1 (Easy)**: small map + strong positive rewards + stronger distance shaping  
  Goal: learn the basic “go to food and eat” behavior
- **Stage 2 (Medium)**: medium map + mild stall/loop penalties  
  Goal: learn to eat consistently without stalling
- **Stage 3 (Hard / Final)**: large map + full stall/loop penalties  
  Goal: learn a more realistic, stable policy

Recommended flow:
1. Train with `STAGE=1` until mean score increases clearly
2. Switch to `STAGE=2` and continue
3. Finish with `STAGE=3`

---

## Checkpoints (so you don’t restart from scratch)
- Auto-saved checkpoints:
  - `checkpoints/stage_1.pt`
  - `checkpoints/stage_2.pt`
  - `checkpoints/stage_3.pt`
- Re-running the same stage will automatically resume from the corresponding checkpoint.

---

## Outputs & Visualization
Every `SHOW_EVERY` episodes:
- prints mean score and epsilon
- prints action distribution (helps detect “always going right” issues)
- runs a demo episode with rendering (pygame)

At the end:
- saves `score_trend.png` and shows the plot window

---

## Known Issues (Future Work)

### 1) Excessive looping
Possible causes:
- reward local optima
- insufficient state/history features

Potential fixes:
- stronger loop detection and trajectory-based penalties
- novelty/curiosity bonuses
- alternative algorithms (e.g., PPO)
- richer observations (direction, relative coordinates, local view)

### 2) Suicidal behavior
Possible causes:
- overly strong negative rewards early on
- sparse success signal and unbalanced termination incentives

Potential fixes:
- time-alive bonus
- reward scaling
- distributional RL variants

---

## Contributing & Roadmap (PRs welcome)
We especially welcome:
- **CPU-only support** (auto device selection, CPU-friendly defaults)
- better anti-loop / anti-suicide mechanisms
- additional algorithms (PPO / Rainbow DQN / C51 / NoisyNet, etc.)
- engineering improvements (TensorBoard, YAML configs, parallel envs)

---
