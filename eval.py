# evalute.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import kymnasium as kym

# 네 코드가 들어있는 파일명에 맞춰 import
# (네가 올린 클래스/함수가 agent.py에 있다고 가정)
from agent import YourBlackAgent, YourWhiteAgent, compute_alive_diff


# ============================================================
# ✅ 여기만 바꿔서 실행
# ============================================================

MODE = "local"          # "local" or "remote"

# --- local 설정 ---
BLACK_WEIGHT = r"C:\Users\pw710\OneDrive\Desktop\Al-Kka-Gi-main (1)\Al-Kka-Gi-main\last.pt"
WHITE_WEIGHT = r"C:\Users\pw710\OneDrive\Desktop\Al-Kka-Gi-main (1)\Al-Kka-Gi-main\last.pt"

EPISODES = 1
RENDER_MODE = "human"      # None or "human"
BGM = True
SEED_BASE: Optional[int] = 123
FPS = 60.0              # human일 때만 의미있음(0이면 sleep 안함)

# --- remote 설정 ---
USER_ID = "YOUR_ID"
COLOR = "black"         # "black" or "white"
WEIGHT = r"C:\Users\pw710\OneDrive\Desktop\Al-Kka-Gi-main (1)\Al-Kka-Gi-main\last.pt"
HOST = "127.0.0.1"
PORT = 12345

ENV_ID = "kymnasium/AlKkaGi-3x3-v0"


# ============================================================
# 내부 로직
# ============================================================

def make_env(render_mode: Optional[str], bgm: bool) -> gym.Env:
    return gym.make(
        ENV_ID,
        obs_type="custom",
        render_mode=render_mode,
        bgm=bgm,
    )


def print_episode_result(ep: int, steps: int, alive_diff_black: float) -> None:
    if alive_diff_black > 0:
        winner = "BLACK"
    elif alive_diff_black < 0:
        winner = "WHITE"
    else:
        winner = "DRAW"
    print(f"[EP {ep:03d}] steps={steps} alive_diff(black)={alive_diff_black:+.0f} => {winner}")


def run_local() -> None:
    black_path = str(Path(BLACK_WEIGHT).expanduser())
    white_path = str(Path(WHITE_WEIGHT).expanduser())

    black_agent = YourBlackAgent.load(black_path)
    white_agent = YourWhiteAgent.load(white_path)

    env = make_env(RENDER_MODE, BGM)

    black_wins = draws = white_wins = 0

    try:
        for ep in range(1, EPISODES + 1):
            obs, info = env.reset(seed=None if SEED_BASE is None else SEED_BASE + ep)
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated):
                turn = int(obs["turn"])
                if turn == 0:
                    action = black_agent.act(obs, info)
                else:
                    action = white_agent.act(obs, info)

                obs, _r, terminated, truncated, info = env.step(action)
                steps += 1

                if RENDER_MODE == "human" and FPS > 0:
                    time.sleep(1.0 / FPS)

            alive_diff_black = compute_alive_diff(obs, my_color=0)
            print_episode_result(ep, steps, alive_diff_black)

            if alive_diff_black > 0:
                black_wins += 1
            elif alive_diff_black < 0:
                white_wins += 1
            else:
                draws += 1

    finally:
        env.close()

    print(f"\n[SUMMARY] black_wins={black_wins} draws={draws} white_wins={white_wins}")


def run_remote() -> None:
    color = COLOR.lower()
    weight_path = str(Path(WEIGHT).expanduser())

    if color == "black":
        agent = YourBlackAgent.load(weight_path)
    elif color == "white":
        agent = YourWhiteAgent.load(weight_path)
    else:
        raise ValueError("COLOR must be 'black' or 'white'")

    kym.evaluate_remote(
        user_id=USER_ID,
        agent=agent,
        host=HOST,
        port=int(PORT),
    )


def main() -> None:
    mode = MODE.lower()
    if mode == "local":
        run_local()
    elif mode == "remote":
        run_remote()
    else:
        raise ValueError("MODE must be 'local' or 'remote'")


if __name__ == "__main__":
    main()
