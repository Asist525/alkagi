"""
알까기 대전/관전 평가 스크립트 (human 렌더링)
- 옵션 없이, 파일 상단 변수만 수정해서 실행
- Torch PPO vs TF(v24) 에이전트 직접 대전
"""

# ============================================================
# 🔧 사용자 설정 영역 (여기만 수정)
# ============================================================
BLACK_AGENT = r"pt:C:\Users\pw710\OneDrive\Desktop\Al-Kka-Gi-main (1)\Al-Kka-Gi-main\last.pt"
# TF v24는 반드시 "base path" ( _actor.keras / _critic.keras 자동 로드 )
WHITE_AGENT = "tf:./moka_white_v24_100000"

EPISODES = 10
RENDER_MODE = "human"      # "human" | "rgb_array"
BGM = False
SEED = None
# ============================================================

# ------------------------------------------------------------
# 환경 변수 (TF GPU 끄기)
# ------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ------------------------------------------------------------
# imports
# ------------------------------------------------------------
import gymnasium as gym
import kymnasium as kym

# ------------------------------------------------------------
# 공통: 승패 판정
# ------------------------------------------------------------
def stone_count(obs: dict, color: str) -> int:
    return sum(1 for s in obs[color] if float(s[2]) > 0.5)

def compute_alive_diff(obs: dict, my_color: int) -> float:
    if my_color == 0:
        me = "black"; opp = "white"
    else:
        me = "white"; opp = "black"
    return float(stone_count(obs, me) - stone_count(obs, opp))

# ------------------------------------------------------------
# agent spec 파서
# ------------------------------------------------------------
def parse_agent_spec(spec: str):
    """
    허용:
      - pt:경로
      - tf:경로(base)
      - *.pt
    """
    s = spec.strip()

    if s.lower().startswith(("pt:", "tf:")):
        k, p = s.split(":", 1)
        return k.lower(), p

    if s.lower().endswith(".pt"):
        return "pt", s

    raise ValueError(f"agent 타입 판별 불가: {spec}")

# ------------------------------------------------------------
# 에이전트 로더 (핵심)
# ------------------------------------------------------------
def load_agent(spec: str, color: int):
    kind, path = parse_agent_spec(spec)

    if kind == "pt":
        # Torch PPO (agent.py)
        from agent import YourBlackAgent, YourWhiteAgent
        return YourBlackAgent.load(path) if color == 0 else YourWhiteAgent.load(path)

    if kind == "tf":
        # TF v24 (alkkagi_v24.py)
        from alkkagi_v24 import BlackAgent, WhiteAgent
        return BlackAgent.load(path) if color == 0 else WhiteAgent.load(path)

    raise ValueError(kind)

# ------------------------------------------------------------
# 평가 루프
# ------------------------------------------------------------
def evaluate_human():
    env = gym.make(
        "kymnasium/AlKkaGi-3x3-v0",
        render_mode=RENDER_MODE,
        obs_type="custom",
        bgm=BGM,
    )

    agent_black = load_agent(BLACK_AGENT, color=0)
    agent_white = load_agent(WHITE_AGENT, color=1)

    bw = ww = dr = 0

    for ep in range(1, EPISODES + 1):
        obs, info = env.reset(seed=None if SEED is None else SEED + ep)
        done = False

        while not done:
            if obs["turn"] == 0:
                action = agent_black.act(obs, info)
            else:
                action = agent_white.act(obs, info)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        diff = compute_alive_diff(obs, my_color=0)
        if diff > 0:
            bw += 1; winner = "Black"
        elif diff < 0:
            ww += 1; winner = "White"
        else:
            dr += 1; winner = "Draw"

        print(f"[EP {ep:03d}/{EPISODES}] winner={winner} | alive_diff={diff:+.0f}")

    env.close()
    print(f"\n[SUMMARY] Black {bw} | White {ww} | Draw {dr}")

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    evaluate_human()
