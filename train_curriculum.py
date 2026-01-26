#!/usr/bin/env python3
"""train_curriculum.py

课程学习训练脚本（最终方案版本）

关键设计：
- **最终 success 固定为 lift+稳定保持（由 env 内部判定）**，success 定义不随课程变化。
- curriculum 仅用于：
  1) 注入训练进度到 env（set_training_progress）
  2) env 内部按 progress 门控/调度 shaping reward
  3) env 内部按 progress 自动提升难度（物体随机范围）

额外改进：
- 修复 "总训练步数 total_timesteps" 不一致导致 progress 计算错误的问题。
- 将 env 的 reward 分解项从 info 中记录到 TensorBoard（rewards/*）。

注意：本脚本假设你的环境类位于 `envs/robotiq_env.py`：
    from envs.robotiq_env import Robotiq2F85Env
如果你把文件放在其他路径，请相应修改 import。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import yaml
import numpy as np
import imageio.v2 as imageio

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.robotiq_env import Robotiq2F85Env


@dataclass
class Paths:
    model_dir: str = "./models/robotiq/"
    log_dir: str = "./logs/robotiq/"
    checkpoint_prefix: str = "sac_robotiq"


class CurriculumCallback(BaseCallback):
    """按训练进度注入 env.set_training_progress(progress)"""

    def __init__(self, total_timesteps: int, update_every_steps: int = 500, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = float(max(1, int(total_timesteps)))
        self.update_every_steps = int(max(1, update_every_steps))
        self.last_phase_id: Optional[int] = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_every_steps != 0:
            return True

        progress = float(self.num_timesteps) / self.total_timesteps
        progress = float(np.clip(progress, 0.0, 1.0))

        # broadcast to all envs
        try:
            num_envs = int(getattr(self.training_env, "num_envs", 1))
            for i in range(num_envs):
                self.training_env.env_method("set_training_progress", progress, indices=[i])
        except Exception as e:
            if self.verbose:
                print(f"[Curriculum] ❌ 更新 set_training_progress 失败: {e}")

        # phase logging (use env 0)
        phase_map = {"approach": 0, "contact": 1, "grasp": 2, "lift": 3}
        try:
            stage_info = self.training_env.env_method("get_current_stage", indices=[0])[0]
            phase = stage_info.get("curriculum_phase", "approach")
            phase_id = phase_map.get(phase, -1)
        except Exception:
            phase_id = -1

        if phase_id != self.last_phase_id:
            if self.verbose:
                print(f"\n🎓 curriculum phase -> {phase_id} (progress={progress:.1%})")
            self.last_phase_id = phase_id

        self.logger.record("curriculum/progress", progress)
        self.logger.record("curriculum/phase_id", phase_id)
        return True


class InfoLoggerCallback(BaseCallback):
    """把 env info 中的分解项写入 TensorBoard。

    你的 env（robotiq_env.py）在 info 里提供了：
      reward_dist/reward_near/reward_grasp/reward_contact/reward_lift/reward_success/reward_penalty
      dist_3d/height_gain/success_hold 等
    """

    def __init__(self, log_every_steps: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_every_steps = int(max(1, log_every_steps))

        self.reward_keys = [
            "reward_dist",
            "reward_near",
            "reward_grasp",
            "reward_contact",
            "reward_lift",
            "reward_success",
            "reward_table",
            "reward_oob",
            "reward_drop",
            "reward_penalty",
        ]
        self.metric_keys = [
            "dist_3d",
            "height_gain",
            "success_hold",
            "left_contact",
            "right_contact",
            "left_table_contact",
            "right_table_contact",
        ]

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every_steps != 0:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True

        # mean across envs (only numeric keys)
        for k in self.reward_keys:
            vals = [info.get(k) for info in infos if isinstance(info.get(k, None), (int, float, np.number))]
            if vals:
                self.logger.record(f"rewards/{k}", float(np.mean(vals)))

        for k in self.metric_keys:
            vals = [info.get(k) for info in infos]
            # booleans -> mean(0/1)
            if vals and isinstance(vals[0], (bool, np.bool_)):
                self.logger.record(f"metrics/{k}", float(np.mean([1.0 if v else 0.0 for v in vals])))
            elif vals and isinstance(vals[0], (int, float, np.number)):
                self.logger.record(f"metrics/{k}", float(np.mean(vals)))

        return True


def make_env(config: Dict[str, Any], render_mode: str = "None"):
    """SubprocVecEnv 工厂函数"""

    def _init():
        env_cfg = config.get("env", {})
        max_episode_steps = int(env_cfg.get("max_episode_steps", 200))
        env = Robotiq2F85Env(render_mode=render_mode, max_episode_steps=max_episode_steps, config=config)
        env = Monitor(env)
        return env

    return _init


def record_video(model_path: str, vecnorm_path: str, config: Dict[str, Any], video_path: str, num_episodes: int = 5):
    """录制模型演示视频"""

    print(f"\n🎥 正在录制视频到: {video_path} ...")

    env = DummyVecEnv([make_env(config, render_mode="rgb_array")])

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = SAC.load(model_path, env=env)

    writer = imageio.get_writer(video_path, fps=30)

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            frame = env.render()
            writer.append_data(frame)

    writer.close()
    env.close()
    print("✅ 视频录制完成！")


def train_curriculum(config_path: str = "config_robotiq.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- read config ----
    train_cfg = config.get("train_params", {})
    model_cfg = config.get("model_params", {})
    vec_cfg = config.get("vecnormalize", {})
    policy_cfg = config.get("policy_kwargs", {})
    paths_cfg = config.get("paths", {})
    callbacks_cfg = config.get("callbacks", {})
    curriculum_cfg = config.get("curriculum", {})

    total_timesteps = int(train_cfg.get("total_timesteps", 2_000_000))
    num_envs = int(train_cfg.get("num_envs", 8))

    paths = Paths(
        model_dir=str(paths_cfg.get("model_dir", "./models/robotiq_curriculum/")),
        log_dir=str(paths_cfg.get("log_dir", "./logs/robotiq_curriculum/")),
        checkpoint_prefix=str(paths_cfg.get("checkpoint_prefix", "sac_robotiq")),
    )

    os.makedirs(paths.model_dir, exist_ok=True)
    os.makedirs(paths.log_dir, exist_ok=True)

    # ---- build env ----
    env = SubprocVecEnv([make_env(config, render_mode="None") for _ in range(num_envs)])

    # VecNormalize：推荐只做 obs 归一化，reward 不归一化（更稳）
    env = VecNormalize(
        env,
        norm_obs=bool(vec_cfg.get("norm_obs", True)),
        norm_reward=bool(vec_cfg.get("norm_reward", False)),
        clip_obs=float(vec_cfg.get("clip_obs", 10.0)),
    )

    # ---- build model ----
    ent_coef = model_cfg.get("ent_coef", "auto")
    if isinstance(ent_coef, str) and ent_coef.lower() != "auto":
        # allow numeric string
        ent_coef = float(ent_coef)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=float(model_cfg.get("learning_rate", 3e-4)),
        buffer_size=int(model_cfg.get("buffer_size", 1_000_000)),
        batch_size=int(model_cfg.get("batch_size", 256)),
        gamma=float(model_cfg.get("gamma", 0.99)),
        tau=float(model_cfg.get("tau", 0.005)),
        ent_coef=ent_coef,
        tensorboard_log=paths.log_dir,
        policy_kwargs=policy_cfg or {"net_arch": [256, 256]},
        verbose=1,
    )

    # ---- callbacks ----
    update_every = int(curriculum_cfg.get("update_every_steps", 500))
    curriculum_cb = CurriculumCallback(total_timesteps=total_timesteps, update_every_steps=update_every, verbose=1)
    info_logger_cb = InfoLoggerCallback(log_every_steps=update_every, verbose=0)

    checkpoint_freq = int(callbacks_cfg.get("checkpoint_freq", 50_000))
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=paths.model_dir,
        name_prefix=paths.checkpoint_prefix,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callback = CallbackList([curriculum_cb, info_logger_cb, checkpoint_cb])

    # ---- train ----
    print(f"\n🚀 开始课程学习训练: total_timesteps={total_timesteps:,}  num_envs={num_envs}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name="SAC_Robotiq_FinalScheme",
        log_interval=int(callbacks_cfg.get("log_interval", 10)),
    )

    # ---- save final ----
    ts = int(time.time())
    model_path = os.path.join(paths.model_dir, f"final_{total_timesteps}_{ts}")
    vecnorm_path = os.path.join(paths.model_dir, f"vecnormalize_{total_timesteps}_{ts}.pkl")

    model.save(model_path)
    env.save(vecnorm_path)
    env.close()

    print(f"\n✅ 模型已保存: {model_path}")
    print(f"✅ VecNormalize 已保存: {vecnorm_path}")

    # ---- record video ----
    os.makedirs("./videos", exist_ok=True)
    video_file = f"./videos/robotiq_eval_{total_timesteps}_{ts}.mp4"
    record_video(model_path, vecnorm_path, config, video_file)


if __name__ == "__main__":
    train_curriculum()