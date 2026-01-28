#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持：评测多回合 + （可选）保存视频（mp4）
- env 使用 render_mode="rgb_array"
- 每步：拿 frame -> (可选) imageio 写入 mp4
"""

import os
import time
from pathlib import Path
import yaml

import imageio.v2 as imageio  # pip install imageio imageio-ffmpeg

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

os.environ.setdefault("MUJOCO_GL", "glfw")

from envs.robotiq_env import Robotiq2F85Env


# ==============================
# 常用参数
# ==============================
MODEL_DIR = Path("./models/robotiq_curriculum")
CONFIG_PATH = Path("config_robotiq.yaml")

# 运行多少回合（抓取多少次）
NUM_EPISODES = 10

# ✅ 是否生成视频（单文件，覆盖所有回合）
ENABLE_RECORD = False

# 录制参数
VIDEO_DIR = Path("./videos")
FPS = 30                 # 输出 mp4 帧率
SLOW_MO_SEC = 0.02       # 每步慢放（全程）
MAX_STEPS_PER_EP = 2000  # 安全上限，防止异常无限跑
# ==============================


def pick_latest(path_dir: Path, pattern: str):
    files = sorted(path_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def safe_reset(vec_env):
    out = vec_env.reset()
    return out[0] if isinstance(out, tuple) else out


def unwrap_for_render(env):
    """尽量拿到底层单环境对象，用于 render()"""
    obj = env
    if hasattr(obj, "venv"):  # VecNormalize
        obj = obj.venv
    if hasattr(obj, "envs") and len(obj.envs) > 0:  # DummyVecEnv
        return obj.envs[0]
    return obj


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"找不到配置文件: {CONFIG_PATH.resolve()}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"找不到模型目录: {MODEL_DIR.resolve()}")

    model_zip = pick_latest(MODEL_DIR, "final_*.zip") or pick_latest(MODEL_DIR, "*.zip")
    if model_zip is None:
        raise FileNotFoundError(f"在 {MODEL_DIR.resolve()} 下找不到任何 .zip 模型文件")

    vecnorm_pkl = pick_latest(MODEL_DIR, "vecnormalize_*.pkl") or pick_latest(MODEL_DIR, "*.pkl")

    print(f"✅ Loading model from: {model_zip}", flush=True)
    if vecnorm_pkl is None:
        print("⚠️ 未找到 VecNormalize .pkl（观测归一化统计），模型表现可能会很差！", flush=True)
    else:
        print(f"✅ Loading VecNormalize from: {vecnorm_pkl}", flush=True)

    # ---- 创建环境：用 rgb_array 满足录制 ----
    config = dict(config)
    config.setdefault("curriculum", {})
    config["curriculum"]["enabled"] = False  # 测试最终能力（不动态门控）

    env_cfg = config.get("env", {})
    base_env = Robotiq2F85Env(
        render_mode="rgb_array",
        max_episode_steps=int(env_cfg.get("max_episode_steps", 200)),
        config=config,
        object_id=env_cfg.get("object_id", "random"),  # 从配置读取物体编号
    )
    base_env.set_curriculum_stage(3)
    base_env.set_training_progress(1.0)

    env = DummyVecEnv([lambda: base_env])

    if vecnorm_pkl is not None and vecnorm_pkl.exists():
        env = VecNormalize.load(str(vecnorm_pkl), env)
        env.training = False
        env.norm_reward = False

    model = SAC.load(str(model_zip), env=env)
    render_env = unwrap_for_render(env)

    writer = None
    video_path = None
    if ENABLE_RECORD:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        video_path = VIDEO_DIR / f"eval_{ts}_episodes_{NUM_EPISODES}.mp4"
        writer = imageio.get_writer(str(video_path), fps=FPS)
        print(f"🎥 视频将保存到: {video_path.resolve()}", flush=True)

    print("=" * 70, flush=True)
    print(f"🤖 开始测试：RECORD={ENABLE_RECORD} | slow={SLOW_MO_SEC}s/step", flush=True)
    print("=" * 70, flush=True)

    try:
        for ep in range(1, NUM_EPISODES + 1):
            ep_t0 = time.perf_counter()

            obs = safe_reset(env)

            total_reward = 0.0
            step_count = 0
            is_success = False
            final_info = {}

            # reset 后先取一帧
            frame = render_env.render()
            if ENABLE_RECORD and writer is not None and frame is not None:
                writer.append_data(frame)

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)

                total_reward += float(rewards[0])
                done = bool(dones[0])
                info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
                final_info = info
                step_count += 1

                frame = render_env.render()
                if ENABLE_RECORD and writer is not None and frame is not None:
                    writer.append_data(frame)

                if SLOW_MO_SEC and SLOW_MO_SEC > 0:
                    time.sleep(SLOW_MO_SEC)

                if done or step_count >= MAX_STEPS_PER_EP:
                    is_success = bool(info.get("is_success", False))
                    break

            ep_elapsed = time.perf_counter() - ep_t0

            print(f"\nEpisode {ep}:", flush=True)
            print(f"  steps: {step_count}", flush=True)
            print(f"  reward: {total_reward:.1f}", flush=True)
            print(f"  success: {'✅ YES' if is_success else '❌ NO'}", flush=True)
            print(f"  phase/stage: {final_info.get('curriculum_phase')}/{final_info.get('difficulty_stage')}", flush=True)
            print(f"  elapsed: {ep_elapsed:.6f} s", flush=True)
            if step_count > 0:
                print(f"  avg_step_wall: {ep_elapsed / step_count:.6f} s/step", flush=True)

    finally:
        if writer is not None:
            writer.close()
        env.close()

    print("\n✅ 完成。", flush=True)
    if video_path is not None:
        print(f"✅ 合并视频: {video_path}", flush=True)


if __name__ == "__main__":
    main()
