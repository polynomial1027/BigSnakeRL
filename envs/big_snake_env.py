from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:
    pygame = None


# 方向：0上 1右 2下 3左
DIRS = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
}


@dataclass
class RenderConfig:
    cell_size: int = 16
    margin: int = 16
    hud_height: int = 48
    fps: int = 30
    grid_alpha: int = 35
    enable_shadows: bool = True
    enable_glow: bool = True


class BigSnakeEnv(gym.Env):
    """
    大型贪吃蛇（ML-friendly）
    - Gymnasium API: reset/step
    - observation: (4, H, W) uint8
      channel0=body channel1=head channel2=food channel3=reserved(walls)
    - action: 0上 1右 2下 3左
    - render: pygame (不读取事件队列，不吞键盘；事件由外部脚本处理)

    Reward 组成：
    1) step_penalty: 每步轻微惩罚
    2) food_reward: 吃到食物奖励
    3) death_penalty: 死亡惩罚
    4) stall penalty: 超过 stall_K 步未吃到食物，逐渐加大惩罚（每步 cap）
    5) loop penalty: 最近窗口内 head 位置重复率过高（绕圈）则惩罚
    6) distance_shaping: 可选，朝食物更近给微弱正奖励（用 dist_prev - dist_now）
    """

    metadata = {"render_modes": ["human", None], "render_fps": 30}

    def __init__(
        self,
        width: int = 40,
        height: int = 30,
        max_steps: int = 5000,
        render_mode: Optional[str] = None,
        forbid_reverse: bool = True,
        # --- core rewards ---
        step_penalty: float = -0.05,
        food_reward: float = 100.0,
        death_penalty: float = -100.0,
        # --- optional distance shaping ---
        distance_shaping: float = 0.0,
        # --- stall penalty ---
        stall_K: int = 100,
        stall_lambda: float = 0.01,
        stall_cap: float = 0.15,
        # --- loop penalty ---
        loop_window: int = 80,
        loop_ratio_threshold: float = 0.40,
        loop_penalty: float = 0.05,
        # --- render config ---
        render_cfg: Optional[RenderConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.w = int(width)
        self.h = int(height)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.forbid_reverse = forbid_reverse

        # reward params
        self.step_penalty = float(step_penalty)
        self.food_reward = float(food_reward)
        self.death_penalty = float(death_penalty)
        self.distance_shaping = float(distance_shaping)

        self.stall_K = int(stall_K)
        self.stall_lambda = float(stall_lambda)
        self.stall_cap = float(stall_cap)

        self.loop_window = int(loop_window)
        self.loop_ratio_threshold = float(loop_ratio_threshold)
        self.loop_penalty = float(loop_penalty)

        self.render_cfg = render_cfg or RenderConfig()
        self._rng = random.Random(seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, self.h, self.w), dtype=np.uint8
        )

        # state
        self.snake: List[Tuple[int, int]] = []
        self.dir: int = 1
        self.food: Tuple[int, int] = (0, 0)
        self.steps: int = 0
        self.score: int = 0

        # trackers
        self._prev_dist: Optional[int] = None
        self.steps_since_food: int = 0
        self.head_history = deque(maxlen=self.loop_window)

        # pygame
        self._pg_inited = False
        self._screen = None
        self._clock = None

    def seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        self.steps = 0
        self.score = 0

        cx, cy = self.w // 2, self.h // 2
        self.dir = 1  # right
        self.snake = [(cx - i, cy) for i in range(4)]  # head at index 0
        self._place_food()

        self.steps_since_food = 0
        self.head_history.clear()
        self.head_history.append(self.snake[0])

        self._prev_dist = self._manhattan(self.snake[0], self.food)

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int):
        self.steps += 1

        # direction constraint
        action = int(action)
        if self.forbid_reverse and self._is_reverse(action, self.dir):
            action = self.dir
        else:
            self.dir = action

        head = self.snake[0]
        dx, dy = DIRS[self.dir]
        new_head = (head[0] + dx, head[1] + dy)

        reward = self.step_penalty
        terminated = False
        truncated = False

        # 默认：又过去了一步没吃到
        self.steps_since_food += 1

        # 撞墙
        if not (0 <= new_head[0] < self.w and 0 <= new_head[1] < self.h):
            reward += self.death_penalty
            terminated = True
        else:
            # 撞自己：若不增长则尾巴会移动，尾巴位置不应算碰撞
            will_grow = (new_head == self.food)
            body_set = set(self.snake[:-1] if not will_grow else self.snake)
            if new_head in body_set:
                reward += self.death_penalty
                terminated = True

        ate_food = False

        if not terminated:
            # move
            self.snake.insert(0, new_head)

            if new_head == self.food:
                ate_food = True
                self.score += 1
                reward += self.food_reward
                self._place_food()
                self.steps_since_food = 0
            else:
                self.snake.pop()

            # update head history
            self.head_history.append(self.snake[0])

            # distance shaping: 朝食物更近则 +，更远则 -
            if self.distance_shaping != 0.0:
                dist = self._manhattan(self.snake[0], self.food)
                if self._prev_dist is not None:
                    reward += self.distance_shaping * (self._prev_dist - dist)
                self._prev_dist = dist

            # stall penalty: 超过 K 步没吃到，线性增长，且 per-step cap
            if (not ate_food) and (self.steps_since_food > self.stall_K) and (self.stall_lambda != 0.0):
                extra = -self.stall_lambda * (self.steps_since_food - self.stall_K) / max(1, self.stall_K)
                extra = max(extra, -abs(self.stall_cap))
                reward += extra

            # loop penalty: 位置重复率过高则惩罚
            if self.loop_penalty != 0.0 and len(self.head_history) == self.loop_window:
                unique_ratio = len(set(self.head_history)) / float(self.loop_window)
                if unique_ratio < self.loop_ratio_threshold:
                    reward -= abs(self.loop_penalty)

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "steps_since_food": self.steps_since_food,
            "unique_ratio": (len(set(self.head_history)) / float(len(self.head_history))) if len(self.head_history) > 0 else 1.0,
        })

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if pygame is None:
            raise RuntimeError("pygame 未安装：pip install pygame")

        if not self._pg_inited:
            pygame.init()
            self._pg_inited = True
            self._clock = pygame.time.Clock()

            cfg = self.render_cfg
            sw = cfg.margin * 2 + self.w * cfg.cell_size
            sh = cfg.margin * 2 + self.h * cfg.cell_size + cfg.hud_height
            self._screen = pygame.display.set_mode((sw, sh))
            pygame.display.set_caption("Big Snake RL Environment")

        cfg = self.render_cfg
        self._clock.tick(cfg.fps)

        self._screen.fill((10, 12, 18))
        self._draw_gradient_bg()
        self._draw_hud()
        self._draw_grid()
        self._draw_food()
        self._draw_snake()

        pygame.display.flip()

        # 关键：不读取事件队列（不 pygame.event.get），避免吞键盘事件
        pygame.event.pump()

    def close(self):
        if self._pg_inited and pygame is not None:
            pygame.quit()
        self._pg_inited = False
        self._screen = None
        self._clock = None

    # ----------------- obs/info -----------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((4, self.h, self.w), dtype=np.uint8)
        for (x, y) in self.snake[1:]:
            obs[0, y, x] = 255  # body
        hx, hy = self.snake[0]
        obs[1, hy, hx] = 255  # head
        fx, fy = self.food
        obs[2, fy, fx] = 255  # food
        # obs[3] reserved (walls)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_len": len(self.snake),
            "head": self.snake[0],
            "food": self.food,
            "dir": self.dir,
        }

    def _place_food(self):
        occupied = set(self.snake)
        while True:
            x = self._rng.randrange(0, self.w)
            y = self._rng.randrange(0, self.h)
            if (x, y) not in occupied:
                self.food = (x, y)
                return

    @staticmethod
    def _is_reverse(a: int, b: int) -> bool:
        return (a + 2) % 4 == b

    @staticmethod
    def _manhattan(p: Tuple[int, int], q: Tuple[int, int]) -> int:
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    # ----------------- drawing -----------------

    def _cell_rect(self, x: int, y: int) -> "pygame.Rect":
        cfg = self.render_cfg
        px = cfg.margin + x * cfg.cell_size
        py = cfg.margin + cfg.hud_height + y * cfg.cell_size
        return pygame.Rect(px, py, cfg.cell_size, cfg.cell_size)

    def _draw_gradient_bg(self):
        if pygame is None:
            return
        w, h = self._screen.get_size()
        for i in range(0, h, 4):
            t = i / max(1, h - 1)
            r = int(10 + 8 * t)
            g = int(12 + 10 * t)
            b = int(18 + 14 * t)
            pygame.draw.rect(self._screen, (r, g, b), pygame.Rect(0, i, w, 4))

    def _draw_hud(self):
        if pygame is None:
            return
        cfg = self.render_cfg
        hud_rect = pygame.Rect(0, 0, self._screen.get_width(), cfg.hud_height)
        pygame.draw.rect(self._screen, (16, 18, 26), hud_rect)

        font = pygame.font.SysFont("Consolas", 20)
        text = f"Score:{self.score}  Steps:{self.steps}  Len:{len(self.snake)}  SinceFood:{self.steps_since_food}"
        surf = font.render(text, True, (220, 225, 235))
        self._screen.blit(surf, (cfg.margin, 12))

    def _draw_grid(self):
        if pygame is None:
            return
        cfg = self.render_cfg
        overlay = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)
        grid_color = (255, 255, 255, cfg.grid_alpha)

        board_x = cfg.margin
        board_y = cfg.margin + cfg.hud_height
        board_w = self.w * cfg.cell_size
        board_h = self.h * cfg.cell_size

        for x in range(self.w + 1):
            px = board_x + x * cfg.cell_size
            pygame.draw.line(overlay, grid_color, (px, board_y), (px, board_y + board_h), 1)
        for y in range(self.h + 1):
            py = board_y + y * cfg.cell_size
            pygame.draw.line(overlay, grid_color, (board_x, py), (board_x + board_w, py), 1)

        self._screen.blit(overlay, (0, 0))

    def _draw_food(self):
        if pygame is None:
            return
        cfg = self.render_cfg
        fx, fy = self.food
        rect = self._cell_rect(fx, fy)

        if cfg.enable_glow:
            glow = pygame.Surface((rect.w * 3, rect.h * 3), pygame.SRCALPHA)
            center = (glow.get_width() // 2, glow.get_height() // 2)
            for radius in range(rect.w, rect.w * 2, 2):
                alpha = max(0, 90 - (radius - rect.w) * 6)
                pygame.draw.circle(glow, (255, 120, 80, alpha), center, radius)
            self._screen.blit(glow, (rect.centerx - center[0], rect.centery - center[1]))

        pygame.draw.rect(self._screen, (255, 120, 80), rect, border_radius=6)
        highlight = pygame.Rect(rect.x + 3, rect.y + 3, rect.w // 3, rect.h // 3)
        pygame.draw.rect(self._screen, (255, 200, 180), highlight, border_radius=4)

    def _draw_snake(self):
        if pygame is None:
            return
        cfg = self.render_cfg

        if cfg.enable_shadows:
            for i, (x, y) in enumerate(reversed(self.snake)):
                rect = self._cell_rect(x, y)
                shadow = rect.move(2, 2)
                alpha = 80 if i == 0 else 55
                pygame.draw.rect(self._screen, (0, 0, 0, alpha), shadow, border_radius=7)

        n = len(self.snake)
        for i, (x, y) in enumerate(self.snake):
            rect = self._cell_rect(x, y)
            t = i / max(1, n - 1)
            r = int(40 * (1 - t) + 20 * t)
            g = int(220 * (1 - t) + 140 * t)
            b = int(180 * (1 - t) + 90 * t)
            pygame.draw.rect(self._screen, (r, g, b), rect, border_radius=7)

            if i == 0:
                self._draw_eyes(rect)

    def _draw_eyes(self, head_rect: "pygame.Rect"):
        if pygame is None:
            return
        cx, cy = head_rect.center
        offset = head_rect.w // 5
        eye_r = max(2, head_rect.w // 10)

        if self.dir == 0:      # up
            p1 = (cx - offset, cy - offset)
            p2 = (cx + offset, cy - offset)
        elif self.dir == 2:    # down
            p1 = (cx - offset, cy + offset)
            p2 = (cx + offset, cy + offset)
        elif self.dir == 1:    # right
            p1 = (cx + offset, cy - offset)
            p2 = (cx + offset, cy + offset)
        else:                  # left
            p1 = (cx - offset, cy - offset)
            p2 = (cx - offset, cy + offset)

        pygame.draw.circle(self._screen, (240, 245, 250), p1, eye_r)
        pygame.draw.circle(self._screen, (240, 245, 250), p2, eye_r)
        pygame.draw.circle(self._screen, (20, 24, 30), p1, max(1, eye_r // 2))
        pygame.draw.circle(self._screen, (20, 24, 30), p2, max(1, eye_r // 2))
