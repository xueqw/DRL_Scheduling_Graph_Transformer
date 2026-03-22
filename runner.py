import numpy as np
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import smoke_test
from sb3_contrib import MaskablePPO

def random_masked_action(env):
    mask = env.action_masks()
    valid_actions = np.flatnonzero(mask)
    return np.random.choice(valid_actions)

def run_episode(env, policy_fn):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    last_info = None

    while not done:
        # 这里的action是一个复合体 是(node_id, loc_choice)的encode_
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info

    # ===== 关键在这里 =====
    step_info = last_info["step_info"]

    # 你根据自己 StepInfo 的字段选一个
    # 常见几种写法（选一个存在的）
    if hasattr(step_info, "makespan"):
        makespan = step_info.makespan
    elif hasattr(step_info, "finish_time"):
        makespan = step_info.finish_time
    else:
        raise ValueError("StepInfo 里没有 makespan / finish_time")

    return total_reward, float(makespan)

def eval_policy(env, policy_fn, n=10):
    rews, ms = [], []
    for _ in range(n):
        r, m = run_episode(env, policy_fn)
        rews.append(r)
        ms.append(m)
    return float(np.mean(rews)), float(np.mean(ms)), float(np.std(ms))

env = smoke_test.set_env()

# random  policy部分
rand = lambda obs: random_masked_action(env)

print("Random:", eval_policy(env, rand, n=10))

# ppo policy部分
env = smoke_test.set_env()
vec_env = DummyVecEnv([lambda: env])
ppo_model = MaskablePPO.load("ppo_dto.zip", env=vec_env)

mean_reward, std_reward = evaluate_policy(
    ppo_model,
    vec_env,
    n_eval_episodes=10,
    deterministic=True,
)

print(f"PPO: mean_reward={mean_reward:.3f}, std_reward={std_reward:.3f}")