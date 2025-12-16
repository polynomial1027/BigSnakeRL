import os
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from envs.big_snake_env import BigSnakeEnv, RenderConfig


# =========================
# 可编辑配置
# =========================
TOT_EPISODES = 20000
SHOW_EVERY = 1                # 每隔多少episode打印+演示（可改）
MAX_STEPS_PER_EPISODE = 5000

# 演示时不要用纯贪心，否则未训练阶段会“固定一个动作看起来不拐弯”
DEMO_EPS = 0.05                    # 演示时epsilon（可改，0.02~0.1都行）

# 训练更容易学到：距离 shaping（朝食物更近给微弱正奖励）
DIST_SHAPING = 0.002               # 关键：0 表示关闭；建议先用 0.001~0.005

# epsilon-greedy（线性衰减）
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 20000         # 衰减慢一点（重要：大地图更需要慢衰减）

# DQN
GAMMA = 0.99
LR = 2e-4                          # 稍微大一点更快看到变化（可改回1e-4）

REPLAY_SIZE = 200_000
MIN_REPLAY_SIZE = 10_000
BATCH_SIZE = 128

TARGET_UPDATE_EVERY_STEPS = 5000
TRAIN_EVERY_STEPS = 1
GRAD_CLIP_NORM = 10.0

# 可视化
MOVING_AVG_WINDOW = 200
PLOT_SAVE_PATH = "score_trend.png"

# 环境大小
ENV_W, ENV_H = 50, 35


# =========================
# 强制 GPU
# =========================
def get_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：此脚本强制 GPU 运行。")
    return torch.device("cuda")


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


def linear_epsilon(episode: int) -> float:
    if episode >= EPS_DECAY_EPISODES:
        return EPS_END
    frac = episode / float(EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


@torch.no_grad()
def select_action(qnet: nn.Module, obs: np.ndarray, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randint(0, 3)
    x = obs_to_tensor(obs, device)
    q = qnet(x)
    return int(torch.argmax(q, dim=1).item())


def train_step(qnet, target_net, opt, batch, device) -> float:
    s, a, r, s2, d = batch_to_tensors(batch, device)

    q = qnet(s)
    q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        q2 = target_net(s2)
        max_q2 = torch.max(q2, dim=1).values
        y = r + GAMMA * (1.0 - d) * max_q2

    loss = nn.functional.smooth_l1_loss(q_sa, y)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), GRAD_CLIP_NORM)
    opt.step()
    return float(loss.item())


def demo_episode(qnet: nn.Module, device: torch.device, title: str = "") -> int:
    """
    演示一局：用 DEMO_EPS（不是0），保证你能看到“会拐弯/会探索”的动态。
    """
    env = BigSnakeEnv(
        width=ENV_W,
        height=ENV_H,
        render_mode="human",
        max_steps=MAX_STEPS_PER_EPISODE,
        render_cfg=RenderConfig(cell_size=16, fps=30),
        distance_shaping=DIST_SHAPING,
    )
    obs, info = env.reset()
    score = 0
    while True:
        a = select_action(qnet, obs, epsilon=DEMO_EPS, device=device)
        obs, r, terminated, truncated, info = env.step(a)
        score = int(info.get("score", score))
        if terminated or truncated:
            break
    env.close()
    print(f"[DEMO] {title}  score={score}  (demo_eps={DEMO_EPS})")
    return score


def main():
    # 固定种子（可删；保留便于复现）
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = get_cuda_device()
    print("Using device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)

    # 训练环境：不渲染，但开距离 shaping，让早期能学到“朝食物走”
    env = BigSnakeEnv(
        width=ENV_W,
        height=ENV_H,
        render_mode=None,
        max_steps=MAX_STEPS_PER_EPISODE,
        distance_shaping=DIST_SHAPING,
    )

    obs, info = env.reset(seed=seed)
    c, h, w = obs.shape
    n_actions = env.action_space.n

    qnet = QNet(c, h, w, n_actions).to(device)
    target_net = QNet(c, h, w, n_actions).to(device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()

    opt = optim.AdamW(qnet.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)

    episode_scores: List[int] = []
    episode_rewards: List[float] = []
    losses: List[float] = []

    global_step = 0

    for ep in range(TOT_EPISODES):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_score = 0

        epsilon = linear_epsilon(ep)
        show = (ep % SHOW_EVERY == 0)

        # 用于确认“不是永远向右”：统计本局动作分布
        action_hist = np.zeros(n_actions, dtype=np.int64)

        for t in range(MAX_STEPS_PER_EPISODE):
            global_step += 1

            a = select_action(qnet, obs, epsilon=epsilon, device=device)
            action_hist[a] += 1

            obs2, r, terminated, truncated, info2 = env.step(a)
            done = bool(terminated or truncated)

            ep_reward += float(r)
            ep_score = int(info2.get("score", ep_score))

            replay.push(Transition(s=obs, a=a, r=float(r), s2=obs2, done=done))
            obs = obs2

            if len(replay) >= MIN_REPLAY_SIZE and (global_step % TRAIN_EVERY_STEPS == 0):
                batch = replay.sample(BATCH_SIZE)
                loss = train_step(qnet, target_net, opt, batch, device)
                losses.append(loss)

            if global_step % TARGET_UPDATE_EVERY_STEPS == 0:
                target_net.load_state_dict(qnet.state_dict())

            if done:
                break

        episode_scores.append(ep_score)
        episode_rewards.append(ep_reward)

        if show:
            window = min(SHOW_EVERY, len(episode_scores))
            mean_score = float(np.mean(episode_scores[-window:]))
            mean_reward = float(np.mean(episode_rewards[-window:]))
            hist_ratio = action_hist / max(1, action_hist.sum())

            print(
                f"[EP {ep:6d}] eps={epsilon:.4f} | "
                f"mean_score(last {window})={mean_score:.3f} | mean_reward={mean_reward:.3f} | "
                f"replay={len(replay)} | action_ratio={hist_ratio.round(3)}"
            )

            # 演示当前策略（带一点探索，避免“固定向右”的错觉）
            demo_episode(qnet, device, title=f"after ep {ep}")

    env.close()

    # 画 score 趋势
    scores = np.array(episode_scores, dtype=np.float32)

    plt.figure()
    plt.plot(scores, label="score per episode")
    if len(scores) >= MOVING_AVG_WINDOW:
        kernel = np.ones(MOVING_AVG_WINDOW, dtype=np.float32) / MOVING_AVG_WINDOW
        moving = np.convolve(scores, kernel, mode="valid")
        plt.plot(
            np.arange(MOVING_AVG_WINDOW - 1, MOVING_AVG_WINDOW - 1 + len(moving)),
            moving,
            label=f"moving avg ({MOVING_AVG_WINDOW})"
        )

    plt.xlabel("Episode")
    plt.ylabel("Score (food eaten)")
    plt.title("Snake DQN Score Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    print(f"Saved plot to: {os.path.abspath(PLOT_SAVE_PATH)}")
    plt.show()


if __name__ == "__main__":
    main()
