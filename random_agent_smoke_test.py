import numpy as np
from envs.big_snake_env import BigSnakeEnv

def main():
    env = BigSnakeEnv(width=60, height=40, render_mode=None, max_steps=2000)
    obs, info = env.reset(seed=0)
    total = 0.0

    for _ in range(5000):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            obs, info = env.reset()
    env.close()
    print("smoke test ok, total reward:", total)

if __name__ == "__main__":
    main()
