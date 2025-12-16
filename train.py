import os
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from envs.big_snake_env import BigSnakeEnv, RenderConfig


# =========================
# 你主要改这些
# =========================
STAGE = 1                 # 1/2/3：课程学习阶段（改这里决定“先学什么再学什么”）
TOT_EPISODES = 3000       # 每次运行训练多少局（会从 checkpoint 续训，不会从头）
SHOW_EVERY = 1          # 每隔多少局：打印平均得分/epsilon + 演示一局（可改）

CHECKPOINT_DIR = "checkpoints"
PLOT_SAVE_PATH = "score_trend.png"

# 训练循环限制（每局最多走多少步，通常与 env.max_steps一致）
MAX_STEPS_PER_EPISODE = 2000

# DQN 超参数
GAMMA = 0.99
LR = 2e-4

REPLAY_SIZE = 200_000
MIN_REPLAY_SIZE = 10_000
BATCH_SIZE = 128

TRAIN_EVERY_STEPS = 1
TARGET_UPDATE_EVERY_STEPS = 5000
GRAD_CLIP_NORM = 10.0

# 演示时的 epsilon：不要用 0，否则未学会前看起来“固定一个动作”
DEMO_EPS = 0.05

# 绘图：滑动平均窗口
MOVING_AVG_WINDOW = 200


# =========================
# 强制 GPU（不允许 CPU）
# =========================
def get_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：此脚本被设置为强制 GPU 运行。")
    return torch.device("cuda")


# =========================
# Curriculum：阶段配置
# =========================
@dataclass(frozen=True)
class StageConfig:
    width: int
    height: int
    max_steps: int

    food_reward: float
    death_penalty: float
    step_penalty: float

    distance_shaping: float

    stall_K: int
    stall_lambda: float
    stall_cap: float

    loop_window: int
    loop_ratio_threshold: float
    loop_penalty: float


def get_stage_config(stage: int) -> StageConfig:
    # Stage 1：先学会“朝食物走并吃到”
    if stage == 1:
        return StageConfig(
            width=30, height=20, max_steps=1000,
            food_reward=100.0,
            death_penalty=-100.0,
            step_penalty=-0.001,
            distance_shaping=0.2,     # 强引导：接近食物给正反馈

            stall_K=10**9, stall_lambda=0.0, stall_cap=0.0,  # 先关停滞罚
            loop_window=80, loop_ratio_threshold=0.40, loop_penalty=0.0,  # 先关绕圈罚
        )

    # Stage 2：中等地图 + 轻度防拖延
    if stage == 2:
        return StageConfig(
            width=40, height=28, max_steps=2000,
            food_reward=5.0,
            death_penalty=-5.0,
            step_penalty=-0.002,
            distance_shaping=0.01,

            stall_K=200, stall_lambda=0.003, stall_cap=0.05,
            loop_window=100, loop_ratio_threshold=0.38, loop_penalty=0.02,
        )

    # Stage 3：最终大地图 + 完整规则
    if stage == 3:
        return StageConfig(
            width=50, height=35, max_steps=3000,
            food_reward=1.0,
            death_penalty=-1.0,
            step_penalty=-0.005,
            distance_shaping=0.005,

            stall_K=100, stall_lambda=0.01, stall_cap=0.15,
            loop_window=120, loop_ratio_threshold=0.35, loop_penalty=0.05,
        )

    raise ValueError("STAGE must be 1, 2, or 3")


def make_env(cfg: StageConfig, render_mode=None) -> BigSnakeEnv:
    return BigSnakeEnv(
        width=cfg.width,
        height=cfg.height,
        render_mode=render_mode,
        max_steps=cfg.max_steps,

        food_reward=cfg.food_reward,
        death_penalty=cfg.death_penalty,
        step_penalty=cfg.step_penalty,

        distance_shaping=cfg.distance_shaping,

        stall_K=cfg.stall_K,
        stall_lambda=cfg.stall_lambda,
        stall_cap=cfg.stall_cap,

        loop_window=cfg.loop_window,
        loop_ratio_threshold=cfg.loop_ratio_threshold,
        loop_penalty=cfg.loop_penalty,
    )


# =========================
# Replay Buffer
# =========================
@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)


# =========================
# Q Network (CNN)
# =========================
class QNet(nn.Module):
    def __init__(self, c: int, h: int, w: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            flat = out.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        return self.head(z)


def obs_to_tensor(obs_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs_uint8).to(device=device, dtype=torch.float32) / 255.0
    return x.unsqueeze(0)


def batch_to_tensors(batch: List[Transition], device: torch.device):
    s = np.stack([t.s for t in batch], axis=0)
    s2 = np.stack([t.s2 for t in batch], axis=0)
    a = np.array([t.a for t in batch], dtype=np.int64)
    r = np.array([t.r for t in batch], dtype=np.float32)
    d = np.array([t.done for t in batch], dtype=np.float32)

    s_t = torch.from_numpy(s).to(device=device, dtype=torch.float32) / 255.0
    s2_t = torch.from_numpy(s2).to(device=device, dtype=torch.float32) / 255.0
    a_t = torch.from_numpy(a).to(device=device)
    r_t = torch.from_numpy(r).to(device=device)
    d_t = torch.from_numpy(d).to(device=device)
    return s_t, a_t, r_t, s2_t, d_t


def linear_epsilon(ep: int, start_ep: int, eps_start: float, eps_end: float, decay_episodes: int) -> float:
    # 从 start_ep 开始计数，让续训时 epsilon 连续
    t = max(0, ep - start_ep)
    if t >= decay_episodes:
        return eps_end
    frac = t / float(decay_episodes)
    return eps_start + frac * (eps_end - eps_start)


@torch.no_grad()
def select_action(qnet: nn.Module, obs: np.ndarray, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randint(0, 3)
    x = obs_to_tensor(obs, device)
    q = qnet(x)
    return int(torch.argmax(q, dim=1).item())


def train_step_double_dqn(qnet, target_net, opt, batch, device) -> float:
    """
    Double DQN：
    a* = argmax_a Q_online(s',a)
    y  = r + gamma*(1-done)*Q_target(s', a*)
    """
    s, a, r, s2, d = batch_to_tensors(batch, device)

    q = qnet(s)
    q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        a2 = torch.argmax(qnet(s2), dim=1)
        q2 = target_net(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
        y = r + GAMMA * (1.0 - d) * q2

    loss = nn.functional.smooth_l1_loss(q_sa, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), GRAD_CLIP_NORM)
    opt.step()
    return float(loss.item())


def save_checkpoint(path: str, qnet, target_net, opt, episode: int, start_episode: int):
    torch.save({
        "qnet": qnet.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": opt.state_dict(),
        "episode": int(episode),
        "start_episode": int(start_episode),
    }, path)


def load_checkpoint(path: str, qnet, target_net, opt, device: torch.device) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    qnet.load_state_dict(ckpt["qnet"])
    target_net.load_state_dict(ckpt["target_net"])
    opt.load_state_dict(ckpt["optimizer"])
    return ckpt


def demo_episode(qnet: nn.Module, device: torch.device, cfg: StageConfig, title: str):
    env = make_env(cfg, render_mode="human")
    # 演示时渲染配置更舒服
    env.render_cfg = RenderConfig(cell_size=16, fps=30)
    obs, info = env.reset()

    score = 0
    for _ in range(cfg.max_steps):
        a = select_action(qnet, obs, epsilon=DEMO_EPS, device=device)
        obs, r, terminated, truncated, info = env.step(a)
        score = int(info.get("score", score))
        if terminated or truncated:
            break

    env.close()
    print(f"[DEMO] {title} | score={score} | demo_eps={DEMO_EPS}")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"stage_{STAGE}.pt")

    # 固定随机种子（便于复现）
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = get_cuda_device()
    print("Using GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "| torch cuda runtime:", torch.version.cuda)

    cfg = get_stage_config(STAGE)
    print(f"[STAGE {STAGE}] map=({cfg.width}x{cfg.height}) max_steps={cfg.max_steps} "
          f"food={cfg.food_reward} death={cfg.death_penalty} step={cfg.step_penalty} "
          f"dist_shape={cfg.distance_shaping} stallK={cfg.stall_K} loop_pen={cfg.loop_penalty}")

    # 训练环境（不渲染）
    env = make_env(cfg, render_mode=None)

    obs, info = env.reset(seed=seed)
    c, h, w = obs.shape
    n_actions = env.action_space.n

    qnet = QNet(c, h, w, n_actions).to(device)
    target_net = QNet(c, h, w, n_actions).to(device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()

    opt = optim.AdamW(qnet.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)

    # 续训：加载 checkpoint（如果存在）
    ckpt = load_checkpoint(ckpt_path, qnet, target_net, opt, device)
    if ckpt is None:
        start_episode = 0
        print(f"[CKPT] not found: {ckpt_path} (start from scratch)")
    else:
        start_episode = int(ckpt["episode"]) + 1
        print(f"[CKPT] loaded: {ckpt_path} (resume at episode {start_episode})")

    # epsilon schedule（随 stage 更合理）
    if STAGE == 1:
        eps_start, eps_end, eps_decay = 1.0, 0.05, 3000
    elif STAGE == 2:
        eps_start, eps_end, eps_decay = 0.6, 0.05, 4000
    else:
        eps_start, eps_end, eps_decay = 0.4, 0.05, 6000

    episode_scores: List[int] = []
    global_step = 0

    for ep in range(start_episode, start_episode + TOT_EPISODES):
        obs, info = env.reset()
        ep_score = 0
        epsilon = linear_epsilon(ep, start_episode, eps_start, eps_end, eps_decay)

        action_hist = np.zeros(n_actions, dtype=np.int64)

        for t in range(min(MAX_STEPS_PER_EPISODE, cfg.max_steps)):
            global_step += 1
            a = select_action(qnet, obs, epsilon=epsilon, device=device)
            action_hist[a] += 1

            obs2, r, terminated, truncated, info2 = env.step(a)
            done = bool(terminated or truncated)

            replay.push(Transition(s=obs, a=a, r=float(r), s2=obs2, done=done))
            obs = obs2

            ep_score = int(info2.get("score", ep_score))

            if len(replay) >= MIN_REPLAY_SIZE and (global_step % TRAIN_EVERY_STEPS == 0):
                batch = replay.sample(BATCH_SIZE)
                train_step_double_dqn(qnet, target_net, opt, batch, device)

            if global_step % TARGET_UPDATE_EVERY_STEPS == 0:
                target_net.load_state_dict(qnet.state_dict())

            if done:
                break

        episode_scores.append(ep_score)

        if ep % SHOW_EVERY == 0:
            window = min(SHOW_EVERY, len(episode_scores))
            mean_score = float(np.mean(episode_scores[-window:]))
            hist_ratio = (action_hist / max(1, action_hist.sum())).round(3)

            print(f"[EP {ep:6d}] eps={epsilon:.4f} | mean_score(last {window})={mean_score:.3f} "
                  f"| replay={len(replay)} | action_ratio={hist_ratio}")

            # 保存 checkpoint（用于“下次继续训练”）
            save_checkpoint(ckpt_path, qnet, target_net, opt, episode=ep, start_episode=start_episode)
            print(f"[CKPT] saved: {ckpt_path}")

            # 演示一局（每 SHOW_EVERY 次）
            demo_episode(qnet, device, cfg, title=f"stage {STAGE} after ep {ep}")

    env.close()

    # 绘图：score 趋势（本次运行的 scores）
    scores = np.array(episode_scores, dtype=np.float32)

    plt.figure()
    plt.plot(scores, label="score per episode (this run)")
    if len(scores) >= MOVING_AVG_WINDOW:
        kernel = np.ones(MOVING_AVG_WINDOW, dtype=np.float32) / MOVING_AVG_WINDOW
        moving = np.convolve(scores, kernel, mode="valid")
        plt.plot(np.arange(MOVING_AVG_WINDOW - 1, MOVING_AVG_WINDOW - 1 + len(moving)), moving,
                 label=f"moving avg ({MOVING_AVG_WINDOW})")

    plt.xlabel("Episode (this run)")
    plt.ylabel("Score (food eaten)")
    plt.title(f"Snake DQN Score Trend (Stage {STAGE})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    print(f"Saved plot to: {os.path.abspath(PLOT_SAVE_PATH)}")
    plt.show()


if __name__ == "__main__":
    main()
