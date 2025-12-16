import pygame
from envs.big_snake_env import BigSnakeEnv, RenderConfig

KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_RIGHT: 1,
    pygame.K_DOWN: 2,
    pygame.K_LEFT: 3,
}

def main():
    env = BigSnakeEnv(
        width=50,
        height=35,
        render_mode="human",
        render_cfg=RenderConfig(cell_size=16, fps=30),
        max_steps=20000,
        distance_shaping=0.0,
    )

    obs, info = env.reset()
    action = 1  # 初始向右

    running = True
    while running:
        # 事件只在这里处理（不要在 env.render() 里再处理）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_TO_ACTION:
                    action = KEY_TO_ACTION[event.key]
                elif event.key == pygame.K_r:
                    obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
