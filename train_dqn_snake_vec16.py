import os
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv  # 关键：Windows 上避免 spawn / pagefile 风暴

from envs.big_snake_env import BigSnakeEnv, RenderConfig


# =========================
# 实验/保存配置
# =========================
RUN_ID = "run_001"                 # 每次训练改这里：run_002 / run_003 ...
BASE_DIR = "checkpoints"           # checkpoints/<RUN_ID>/...
SAVE_SNAPSHOT_EVERY = 5000         # 每隔多少 episode 额外存一次快照；0=不存

# =========================
# 并行训练配置
# =========================
NUM_ENVS = 1                      # 并行环境数量（SyncVectorEnv 单进程）
TOT_EPISODES = 20000               # 总 episode（跨所有 env 累计）
SHOW_EVERY = 1                  # 每隔多少 episode 打印 + 保存 + 演示
MAX_STEPS_PER_EP_DEMO = 3000       # 演示时单局最多步数

# 地图大小（建议先小一些跑稳，再加大）
ENV_W, ENV_H = 30, 20

# epsilon schedule（按 episode 衰减）
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 20000

# DQN
GAMMA = 0.99
LR = 2e-4

REPLAY_SIZE = 300_000
MIN_REPLAY_SIZE = 20_000
BATCH_SIZE = 256

TARGET_UPDATE_EVERY_STEPS = 5000
TRAIN_EVERY_STEPS = 1
GRAD_CLIP_NORM = 10.0

# 绘图
MOVING_AVG_WINDOW = 200

# 演示 epsilon（不要为 0，避免看起来“永远一个动作”）
DEMO_EPS = 0.05


# =========================
# 强制 GPU
# =========================
def get_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：该脚本强制 GPU 运行。")
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
# Q Network (CNN + MLP head)
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


# =========================
# epsilon schedule
# =========================
def epsilon_by_episode(ep: int) -> float:
    if ep >= EPS_DECAY_EPISODES:
        return EPS_END
    frac = ep / float(EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


# =========================
# tensor helpers
# =========================
def obs_to_tensor(obs_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    # (B,C,H,W) uint8 -> float32 [0,1]
    return torch.from_numpy(obs_uint8).to(device=device, dtype=torch.float32) / 255.0


def sample_actions(qnet: nn.Module, obs: np.ndarray, eps: float, device: torch.device) -> np.ndarray:
    """
    obs: (B,C,H,W) uint8
    """
    B = obs.shape[0]
    actions = np.empty((B,), dtype=np.int64)

    rand_mask = np.random.rand(B) < eps
    if rand_mask.any():
        actions[rand_mask] = np.random.randint(0, 4, size=int(rand_mask.sum()))

    if (~rand_mask).any():
        with torch.no_grad():
            x = obs_to_tensor(obs[~rand_mask], device)
            q = qnet(x)
            a = torch.argmax(q, dim=1).to("cpu").numpy()
        actions[~rand_mask] = a

    return actions


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


def train_step_double_dqn(qnet, target_net, opt, batch, device) -> float:
    """
    Double DQN:
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


# =========================
# checkpoint & metrics
# =========================
def save_ckpt(path: str, qnet, target_net, opt, total_episodes: int, total_steps: int, epsilon: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "qnet": qnet.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": opt.state_dict(),
        "total_episodes": int(total_episodes),
        "total_steps": int(total_steps),
        "epsilon": float(epsilon),
    }, path)


def load_ckpt(path: str, qnet, target_net, opt, device: torch.device):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    qnet.load_state_dict(ckpt["qnet"])
    target_net.load_state_dict(ckpt["target_net"])
    opt.load_state_dict(ckpt["optimizer"])
    return ckpt


def append_metrics_csv(metrics_path: str, episode: int, epsilon: float, mean_score: float, replay_size: int, total_steps: int):
    need_header = (not os.path.exists(metrics_path)) or (os.path.getsize(metrics_path) == 0)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write("episode,epsilon,mean_score,replay_size,total_steps\n")
        f.write(f"{episode},{epsilon:.6f},{mean_score:.6f},{replay_size},{total_steps}\n")


# =========================
# env factory
# =========================
def make_env(seed: int):
    env = BigSnakeEnv(
        width=ENV_W,
        height=ENV_H,
        render_mode=None,
        limit_step=True,
        step_limit_factor=4,
        seed=seed,
        max_steps=10**9,
    )
    return env


def make_env_fn(seed: int):
    def _thunk():
        return make_env(seed)
    return _thunk


# =========================
# info extraction (VectorEnv-safe)
# =========================
def update_scores_from_infos(ep_scores: np.ndarray, infos: Any, done_mask: np.ndarray):
    """
    ep_scores: shape (NUM_ENVS,)
    infos: VectorEnv infos (typically dict-of-arrays)
    done_mask: shape (NUM_ENVS,) bool for terminated|truncated
    """
    if not isinstance(infos, dict):
        return

    # 1) common case: infos["score"] is an array (NUM_ENVS,)
    if "score" in infos:
        try:
            ep_scores[:] = np.asarray(infos["score"], dtype=np.int64)
        except Exception:
            pass

    # 2) gymnasium may provide per-env final info when done:
    #    infos["final_info"] + infos["final_info_mask"]
    if "final_info" in infos and "final_info_mask" in infos:
        try:
            fmask = np.asarray(infos["final_info_mask"], dtype=bool)
        except Exception:
            fmask = None

        finfo = infos["final_info"]

        if fmask is not None and fmask.any():
            # finfo might be list-of-dicts aligned to envs
            if isinstance(finfo, (list, tuple)) and len(finfo) == len(ep_scores):
                for i in np.where(fmask)[0]:
                    if isinstance(finfo[i], dict) and "score" in finfo[i]:
                        ep_scores[i] = int(finfo[i]["score"])


# =========================
# demo
# =========================
def demo_episode(qnet, device: torch.device, title: str = ""):
    env = BigSnakeEnv(
        width=ENV_W,
        height=ENV_H,
        render_mode="human",
        limit_step=True,
        step_limit_factor=4,
        seed=0,
    )
    env.render_cfg = RenderConfig(cell_size=18, fps=30)
    obs, info = env.reset()

    score = 0
    steps = 0
    while True:
        steps += 1
        a = sample_actions(qnet, obs[None, ...], DEMO_EPS, device)[0]
        obs, r, terminated, truncated, info = env.step(int(a))
        score = int(info.get("score", score))
        if terminated or truncated or steps >= MAX_STEPS_PER_EP_DEMO:
            break

    env.close()
    print(f"[DEMO] {title} score={score} (demo_eps={DEMO_EPS})")


# =========================
# main
# =========================
def main():
    # run dir
    run_dir = os.path.join(BASE_DIR, RUN_ID)
    os.makedirs(run_dir, exist_ok=True)

    ckpt_latest_path = os.path.join(run_dir, "ckpt_latest.pt")
    ckpt_final_path = os.path.join(run_dir, "ckpt_final.pt")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    plot_path = os.path.join(run_dir, "score_trend.png")

    device = get_cuda_device()
    print("Using GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
    print(f"[RUN] RUN_ID={RUN_ID} -> {os.path.abspath(run_dir)}")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Vector env (single process)
    env_fns = [make_env_fn(seed=i) for i in range(NUM_ENVS)]
    venv = SyncVectorEnv(env_fns)

    obs, infos = venv.reset()
    if not isinstance(obs, np.ndarray):
        obs = np.asarray(obs)

    B, C, H, W = obs.shape
    n_actions = 4

    qnet = QNet(C, H, W, n_actions).to(device)
    target_net = QNet(C, H, W, n_actions).to(device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()

    opt = optim.AdamW(qnet.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)

    total_episodes_done = 0
    total_steps_done = 0
    last_eps = EPS_START

    ckpt = load_ckpt(ckpt_latest_path, qnet, target_net, opt, device)
    if ckpt is not None:
        total_episodes_done = int(ckpt.get("total_episodes", 0))
        total_steps_done = int(ckpt.get("total_steps", 0))
        last_eps = float(ckpt.get("epsilon", EPS_START))
        print(f"[CKPT] loaded latest: episodes={total_episodes_done} steps={total_steps_done} eps={last_eps:.4f}")
    else:
        print("[CKPT] not found, start from scratch")

    ep_scores = np.zeros((NUM_ENVS,), dtype=np.int64)
    recent_scores = deque(maxlen=SHOW_EVERY)

    logged_eps = []
    logged_mean = []
    logged_ep = []

    # warm-up: try to populate ep_scores from reset infos if available
    update_scores_from_infos(ep_scores, infos, np.zeros((NUM_ENVS,), dtype=bool))

    while total_episodes_done < TOT_EPISODES:
        eps = epsilon_by_episode(total_episodes_done)
        last_eps = eps

        # actions = sample_actions(qnet, obs, eps, device)
        # 重要：replay 没满 MIN_REPLAY_SIZE 前，强制全随机，避免“永远同一个动作”
        effective_eps = 1.0 if len(replay) < MIN_REPLAY_SIZE else eps
        actions = sample_actions(qnet, obs, effective_eps, device)


        next_obs, rewards, terms, truncs, infos = venv.step(actions)
        done = np.logical_or(terms, truncs)

        # push transitions
        for i in range(NUM_ENVS):
            replay.push(Transition(
                s=obs[i],
                a=int(actions[i]),
                r=float(rewards[i]),
                s2=next_obs[i],
                done=bool(done[i]),
            ))

        obs = next_obs
        total_steps_done += NUM_ENVS

        # update scores safely from vector infos
        update_scores_from_infos(ep_scores, infos, done)

        # train
        if len(replay) >= MIN_REPLAY_SIZE and (total_steps_done % TRAIN_EVERY_STEPS == 0):
            batch = replay.sample(BATCH_SIZE)
            train_step_double_dqn(qnet, target_net, opt, batch, device)

        # update target
        if total_steps_done % TARGET_UPDATE_EVERY_STEPS == 0:
            target_net.load_state_dict(qnet.state_dict())

        # handle done envs -> count episodes
        if done.any():
            done_ids = np.where(done)[0]
            for i in done_ids:
                recent_scores.append(int(ep_scores[i]))
                total_episodes_done += 1
                ep_scores[i] = 0

                if total_episodes_done % SHOW_EVERY == 0:
                    mean_score = float(np.mean(recent_scores)) if len(recent_scores) else 0.0
                    print(
                        f"[EP {total_episodes_done:6d}] eps={eps:.4f} | "
                        f"mean_score(last {len(recent_scores)})={mean_score:.3f} | "
                        f"replay={len(replay)} | steps={total_steps_done}"
                    )

                    append_metrics_csv(metrics_path, total_episodes_done, eps, mean_score, len(replay), total_steps_done)

                    # save latest
                    save_ckpt(ckpt_latest_path, qnet, target_net, opt, total_episodes_done, total_steps_done, eps)
                    print(f"[CKPT] saved latest: {ckpt_latest_path}")

                    # optional snapshot
                    if SAVE_SNAPSHOT_EVERY and (total_episodes_done % SAVE_SNAPSHOT_EVERY == 0):
                        snap_path = os.path.join(run_dir, f"ckpt_ep_{total_episodes_done:06d}.pt")
                        save_ckpt(snap_path, qnet, target_net, opt, total_episodes_done, total_steps_done, eps)
                        print(f"[CKPT] saved snapshot: {snap_path}")

                    logged_ep.append(total_episodes_done)
                    logged_eps.append(eps)
                    logged_mean.append(mean_score)

                    # demo
                    demo_episode(qnet, device, title=f"{RUN_ID} after ep {total_episodes_done}")

                if total_episodes_done >= TOT_EPISODES:
                    break

    venv.close()

    # final save
    save_ckpt(ckpt_final_path, qnet, target_net, opt, total_episodes_done, total_steps_done, last_eps)
    print(f"[CKPT] saved final: {ckpt_final_path}")

    # plot
    if len(logged_mean) == 0:
        print("No logged points to plot (increase training or reduce SHOW_EVERY).")
        return

    x = np.array(logged_ep, dtype=np.int32)
    y = np.array(logged_mean, dtype=np.float32)

    plt.figure()
    plt.plot(x, y, label=f"mean_score every {SHOW_EVERY} episodes")

    if len(y) >= 3:
        w = min(MOVING_AVG_WINDOW, len(y))
        kernel = np.ones(w, dtype=np.float32) / w
        moving = np.convolve(y, kernel, mode="valid")
        plt.plot(x[w-1:], moving, label=f"moving avg (w={w})")

    plt.xlabel("Episode")
    plt.ylabel("Mean Score")
    plt.title(f"BigSnakeRL Vec{NUM_ENVS} — {RUN_ID}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {os.path.abspath(plot_path)}")
    plt.show()


if __name__ == "__main__":
    main()
