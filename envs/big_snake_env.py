from __future__ import annotations

import math
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
    BigSnakeEnv with "author-style" reward mechanism (the one you pasted).

    Observation: (4, H, W) uint8
      ch0=body, ch1=head, ch2=food, ch3=reserved
    Actions: Discrete(4) 0 up,1 right,2 down,3 left

    Reward (author-style):
      - Victory: 0.1 * max_growth
      - Game over: -0.1 * max_growth^((grid_size - snake_size)/max_growth)  in (-0.1*max_growth, -0.1]
      - Food: snake_size / grid_size   (0..1)
      - Shaping: +0.1*(1/snake_size) if closer to food else -0.1*(1/snake_size)
      - Step limit: step_limit = grid_size * 4 (default) triggers done (then game over penalty)

    Notes:
      - forbid_reverse=True prevents direct reverse moves.
      - render() uses pygame; training should run with render_mode=None.
    """

    metadata = {"render_modes": ["human", None], "render_fps": 30}

    def __init__(
        self,
        width: int = 40,
        height: int = 30,
        max_steps: int = 10**9,          # keep, but step_limit will end earlier if enabled
        render_mode: Optional[str] = None,
        forbid_reverse: bool = True,

        # author-style toggles
        limit_step: bool = True,
        step_limit_factor: int = 4,      # step_limit = grid_size * factor

        # render
        render_cfg: Optional[RenderConfig] = None,

        seed: Optional[int] = None,
    ):
        super().__init__()
        self.w = int(width)
        self.h = int(height)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.forbid_reverse = forbid_reverse

        self.limit_step = bool(limit_step)
        self.step_limit_factor = int(step_limit_factor)

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

        # author-style counters
        self.grid_size = self.w * self.h
        self.init_snake_size = 4
        self.max_growth = self.grid_size - self.init_snake_size
        self.step_limit = (self.grid_size * self.step_limit_factor) if self.limit_step else int(1e12)
        self.reward_step_counter = 0

        # rendering
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
        self.reward_step_counter = 0

        cx, cy = self.w // 2, self.h // 2
        self.dir = 1  # right
        self.snake = [(cx - i, cy) for i in range(self.init_snake_size)]
        self._place_food()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int):
        self.steps += 1

        action = int(action)
        if self.forbid_reverse and self._is_reverse(action, self.dir):
            action = self.dir
        else:
            self.dir = action

        self.reward_step_counter += 1

        head = self.snake[0]
        dx, dy = DIRS[self.dir]
        new_head = (head[0] + dx, head[1] + dy)

        prev_dist = self._l2(head, self.food)

        done = False
        truncated = False
        reward = 0.0

        # victory (fills board)
        if len(self.snake) == self.grid_size:
            reward = self.max_growth * 0.1
            done = True
            obs = self._get_obs()
            info = self._get_info()
            return obs, float(reward), done, truncated, info

        # step limit (author-style)
        if self.reward_step_counter > self.step_limit:
            self.reward_step_counter = 0
            done = True

        # compute collisions (tail logic)
        will_grow = (new_head == self.food)

        # wall collision
        if not (0 <= new_head[0] < self.w and 0 <= new_head[1] < self.h):
            done = True
        else:
            if will_grow:
                body_set = set(self.snake)           # tail not removed
            else:
                body_set = set(self.snake[:-1])      # tail would move
            if new_head in body_set:
                done = True

        # if done => game over penalty (author-style, based on snake size)
        if done:
            snake_size = len(self.snake)
            exp = (self.grid_size - snake_size) / max(1, self.max_growth)
            reward = - math.pow(max(1.0, float(self.max_growth)), exp)
            reward *= 0.1
            obs = self._get_obs()
            info = self._get_info()
            return obs, float(reward), True, truncated, info

        # not done => perform move
        self.snake.insert(0, new_head)

        food_obtained = False
        if will_grow:
            food_obtained = True
            self.score += 1
            self._place_food()
            self.reward_step_counter = 0
        else:
            self.snake.pop()

        # food reward (author-style)
        if food_obtained:
            reward = len(self.snake) / float(self.grid_size)
        else:
            # tiny shaping based on closer/farther to food
            new_dist = self._l2(new_head, self.food)
            if new_dist < prev_dist:
                reward = 1.0 / float(len(self.snake))
            else:
                reward = -1.0 / float(len(self.snake))
            reward *= 0.1

        # truncate by max_steps if you still want a hard cap
        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info["food_obtained"] = food_obtained
        info["reward_step_counter"] = self.reward_step_counter
        info["step_limit"] = self.step_limit

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), False, truncated, info

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
            obs[0, y, x] = 255
        hx, hy = self.snake[0]
        obs[1, hy, hx] = 255
        fx, fy = self.food
        obs[2, fy, fx] = 255
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_size": len(self.snake),
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
    def _l2(p: Tuple[int, int], q: Tuple[int, int]) -> float:
        # L2 distance, same spirit as the reference code
        return float(np.linalg.norm(np.array(p, dtype=np.float32) - np.array(q, dtype=np.float32)))

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
        text = f"Score:{self.score}  Steps:{self.steps}  Len:{len(self.snake)}  Ctr:{self.reward_step_counter}/{self.step_limit}"
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

        if self.dir == 0:
            p1 = (cx - offset, cy - offset)
            p2 = (cx + offset, cy - offset)
        elif self.dir == 2:
            p1 = (cx - offset, cy + offset)
            p2 = (cx + offset, cy + offset)
        elif self.dir == 1:
            p1 = (cx + offset, cy - offset)
            p2 = (cx + offset, cy + offset)
        else:
            p1 = (cx - offset, cy - offset)
            p2 = (cx - offset, cy + offset)

        pygame.draw.circle(self._screen, (240, 245, 250), p1, eye_r)
        pygame.draw.circle(self._screen, (240, 245, 250), p2, eye_r)
        pygame.draw.circle(self._screen, (20, 24, 30), p1, max(1, eye_r // 2))
        pygame.draw.circle(self._screen, (20, 24, 30), p2, max(1, eye_r // 2))
