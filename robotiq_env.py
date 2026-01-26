"""robotiq_env.py

Robotiq 2F-85 抓取环境（MuJoCo）

本版本按“最终方案”重构：
- **最终 success（is_success）固定为 lift 并保持稳定 N 步**，success 定义不随课程变化。
- 课程学习（curriculum）仅用于：
  - reward 的门控与权重调度（shaping），避免目标函数非平稳；
  - 难度（物体初始随机范围）的自动提升；
  - 记录阶段信息（TensorBoard/日志）。
- **所有 shaping 奖励均避免“每步可刷正分”**：
  - 距离：用 potential-based shaping（变近才给分）。
  - near/contact：事件奖励（每回合最多触发一次）。
  - grasp：奖励“闭合动作增量”，不是奖励“闭合状态”。
  - lift：奖励“高度增量”，不是奖励“已抬起高度”。
- 增加 pad-桌面接触惩罚，增加 mocap workspace clamp（尤其 z 方向），抑制压桌刷分。

与 stable-baselines3 (Gymnasium API) 兼容：step() 返回 (obs, reward, terminated, truncated, info)
"""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, Tuple, Optional

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


class Robotiq2F85Env(gym.Env):
    """Robotiq 2F-85 夹爪抓取环境（6DoF mocap 控制 + 夹爪开合）"""

    metadata = {"render_modes": ["None", "human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str = "None",
        render_every: int = 1,
        max_episode_steps: int = 200,
        config: Optional[dict] = None,
    ):
        super().__init__()

        self.config = config or {}
        self.render_mode = render_mode
        self.render_every = int(render_every)
        self._render_counter = 0

        # RNG
        seed = int(self.config.get("seed", 0)) if isinstance(self.config, dict) else 0
        self.np_random = np.random.RandomState(seed)

        # ---- Load model ----
        self.model, self.data = self._load_model_from_config(self.config)

        # Per-action physics steps
        env_cfg = self.config.get("env", {})
        self.physics_steps_per_action = int(env_cfg.get("physics_steps_per_action", 10))
        self.max_episode_steps = int(env_cfg.get("max_episode_steps", max_episode_steps))

        # ---- Cache IDs ----
        def safe_name2id(objtype, name: str) -> int:
            try:
                return int(mujoco.mj_name2id(self.model, objtype, name))
            except Exception:
                return -1

        # joints
        self.right_driver_joint = safe_name2id(mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        self.left_driver_joint = safe_name2id(mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")

        # bodies
        self.target_body_id = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self.table_body_id = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "table")
        self.gripper_base_id = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self.right_pad_body = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "right_pad")
        self.left_pad_body = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "left_pad")

        # geoms
        self.target_geom_id = safe_name2id(mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.table_geom_id = safe_name2id(mujoco.mjtObj.mjOBJ_GEOM, "table_top")

        # sites
        self.pinch_site_id = safe_name2id(mujoco.mjtObj.mjOBJ_SITE, "pinch")

        # mocap body
        hand_mocap_body_id = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "hand_mocap")
        if hand_mocap_body_id >= 0:
            self.gripper_mocap_id = int(self.model.body_mocapid[hand_mocap_body_id])
        else:
            self.gripper_mocap_id = -1

        # qpos indices
        self.gripper_qpos_start = None
        gripper_joint_id = safe_name2id(mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")
        if gripper_joint_id >= 0:
            self.gripper_qpos_start = int(self.model.jnt_qposadr[gripper_joint_id])
            self.gripper_dof_start = int(self.model.jnt_dofadr[gripper_joint_id])
        else:
            self.gripper_dof_start = None

        self.target_qpos_start = None
        target_joint_id = safe_name2id(mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
        if target_joint_id >= 0:
            self.target_qpos_start = int(self.model.jnt_qposadr[target_joint_id])

        # driver qpos
        self.right_driver_qpos = int(self.model.jnt_qposadr[self.right_driver_joint]) if self.right_driver_joint >= 0 else None
        self.left_driver_qpos = int(self.model.jnt_qposadr[self.left_driver_joint]) if self.left_driver_joint >= 0 else None

        # ---- Table/object geometry derived constants ----
        mujoco.mj_forward(self.model, self.data)
        self.table_top_z = self._compute_table_top_z()
        self.object_half_z = float(self.model.geom_size[self.target_geom_id][2]) if self.target_geom_id >= 0 else 0.02

        # ---- Control config ----
        control_cfg = self.config.get("control", {})
        self.position_scale = float(control_cfg.get("position_scale", 0.02))
        self.gripper_max_delta = float(control_cfg.get("gripper_max_delta", 10.0))

        # Workspace clamp (mocap)
        # x/y 默认按桌子范围 clamp（table_top size = 0.3），可在 config.control 覆盖
        default_xy = 0.30
        if self.table_geom_id >= 0:
            default_xy = float(self.model.geom_size[self.table_geom_id][0])
        self.x_min = float(control_cfg.get("x_min", -default_xy))
        self.x_max = float(control_cfg.get("x_max", default_xy))
        self.y_min = float(control_cfg.get("y_min", -default_xy))
        self.y_max = float(control_cfg.get("y_max", default_xy))

        # z_min/z_max：强制至少高于桌面（避免压桌/穿透）
        z_min_cfg = float(control_cfg.get("z_min", 0.05))
        z_max_cfg = float(control_cfg.get("z_max", 0.60))
        # 允许用户显式设置更高的 z_min；若太低则抬到桌面上方 margin
        self.z_table_margin = float(control_cfg.get("z_table_margin", 0.01))
        self.z_min = max(z_min_cfg, self.table_top_z + self.z_table_margin)
        self.z_max = max(self.z_min + 1e-3, z_max_cfg)

        # ---- Reward config ----
        reward_cfg = self.config.get("reward_weights", {})
        self.base_reward_weights = {
            "distance": float(reward_cfg.get("distance", 1.0)),
            "near_object": float(reward_cfg.get("near_object", 2.0)),
            "grasp_guidance": float(reward_cfg.get("grasp_guidance", 1.0)),
            "single_contact": float(reward_cfg.get("single_contact", 2.0)),
            "both_contact": float(reward_cfg.get("both_contact", 5.0)),
            "lift": float(reward_cfg.get("lift", 50.0)),
            "success": float(reward_cfg.get("success", 200.0)),
            "time_penalty": float(reward_cfg.get("time_penalty", -0.01)),
            "drop_penalty": float(reward_cfg.get("drop_penalty", -50.0)),
            "table_contact_penalty": float(reward_cfg.get("table_contact_penalty", -0.5)),
            "out_of_bounds_penalty": float(reward_cfg.get("out_of_bounds_penalty", -10.0)),
        }
        # Active weights (after curriculum multipliers)
        self.reward_weights = self.base_reward_weights.copy()

        # ---- Reward shaping parameters ----
        reward_params = self.config.get("reward_params", {})
        self.dist_alpha = float(reward_params.get("dist_alpha", 5.0))
        self.near_threshold = float(reward_params.get("near_threshold", 0.15))
        self.align_xy_thresh = float(reward_params.get("align_xy_thresh", 0.03))
        self.align_z_thresh = float(reward_params.get("align_z_thresh", 0.05))

        # ---- Success criteria (FINAL ONLY) ----
        success_cfg = self.config.get("success_criteria", {})
        self.min_lift_height = float(success_cfg.get("min_lift_height", 0.02))
        self.require_both_contact = bool(success_cfg.get("require_both_contact", True))
        self.hold_steps = int(success_cfg.get("hold_steps", 10))

        # ---- Curriculum ----
        curriculum_cfg = self.config.get("curriculum", {})
        self.curriculum_enabled = bool(curriculum_cfg.get("enabled", True))
        self.auto_difficulty = bool(curriculum_cfg.get("auto_difficulty", True))
        # phase schedule in progress fraction
        self.phase_bins = curriculum_cfg.get(
            "phase_bins",
            {
                "approach": [0.0, 0.25],
                "contact": [0.25, 0.50],
                "grasp": [0.50, 0.75],
                "lift": [0.75, 1.01],
            },
        )
        self.training_progress = 0.0
        self.curriculum_phase = "approach"

        # manual and auto difficulty stage
        self.curriculum_stage = 0  # manual floor stage (set_curriculum_stage)
        self._auto_stage = 0

        # ---- Spaces ----
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        # ---- Runtime state ----
        self.steps = 0
        self.viewer = None
        self._renderer = None

        # object height baseline (set in reset)
        self.initial_obj_height = 0.0

        # gripper control (0=open, 255=closed)
        self.current_gripper_ctrl = 0.0

        # success tracking (final)
        self.success_hold = 0
        self._success_ever = False

        # shaping state
        self.prev_dist_3d: Optional[float] = None
        self.prev_obj_height: Optional[float] = None
        self.prev_ctrl: float = 0.0
        self.near_once = False
        self.single_contact_once = False
        self.both_contact_once = False

        # stats
        self.success_history = deque(maxlen=100)

        # apply initial curriculum weights
        self._update_curriculum()

    # ---------------------------------------------------------------------
    # Model loading / geometry utils
    # ---------------------------------------------------------------------

    def _load_model_from_config(self, config: dict) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Load MuJoCo model path.

        Priority:
        1) config['env']['model_path'] if exists
        2) default: ../mujoco_menagerie/robotiq_2f85/grasping_scene.xml
        """
        env_cfg = config.get("env", {}) if isinstance(config, dict) else {}
        model_path_cfg = env_cfg.get("model_path", None)

        candidates = []
        if model_path_cfg:
            if os.path.isabs(model_path_cfg):
                candidates.append(model_path_cfg)
            else:
                # try relative to this file
                candidates.append(os.path.join(os.path.dirname(__file__), model_path_cfg))
                # try relative to repo root (one level up)
                candidates.append(os.path.join(os.path.dirname(__file__), "..", model_path_cfg))

        # default fallback
        default_model_dir = os.path.join(os.path.dirname(__file__), "..", "mujoco_menagerie", "robotiq_2f85")
        candidates.append(os.path.join(default_model_dir, "grasping_scene.xml"))

        model_path = None
        for p in candidates:
            if p and os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            raise FileNotFoundError(
                f"Model file not found. Tried: {candidates}. "
                "Please set config['env']['model_path'] correctly."
            )

        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        return model, data

    def _compute_table_top_z(self) -> float:
        if self.table_body_id < 0:
            return 0.30
        # world z of table body
        z_body = float(self.data.xpos[self.table_body_id][2])
        # max geom half-height on table body
        max_half_z = 0.0
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) == int(self.table_body_id):
                max_half_z = max(max_half_z, float(self.model.geom_size[geom_id][2]))
        if max_half_z <= 0.0 and self.table_geom_id >= 0:
            max_half_z = float(self.model.geom_size[self.table_geom_id][2])
        return z_body + max_half_z

    # ---------------------------------------------------------------------
    # Curriculum
    # ---------------------------------------------------------------------

    def set_curriculum_stage(self, stage: int):
        """手动设置难度 stage（0-3）。训练时会与 auto stage 取 max。"""
        self.curriculum_stage = int(np.clip(stage, 0, 3))

    def set_training_progress(self, progress: float):
        """由 callback 注入训练进度 (0-1)。"""
        self.training_progress = float(np.clip(progress, 0.0, 1.0))
        if self.curriculum_enabled:
            self._update_curriculum()

    def _update_curriculum(self):
        """根据 training_progress 更新：
        - curriculum_phase（用于门控/权重）
        - reward_weights（base 权重乘以 multiplier）
        - auto difficulty stage（影响 reset 随机范围）
        """
        p = float(np.clip(self.training_progress, 0.0, 1.0))

        # phase
        phase = "approach"
        for name, (a, b) in self.phase_bins.items():
            if a <= p < b:
                phase = name
                break
        self.curriculum_phase = phase

        # difficulty stage
        if self.auto_difficulty:
            if p < 0.25:
                self._auto_stage = 0
            elif p < 0.50:
                self._auto_stage = 1
            elif p < 0.75:
                self._auto_stage = 2
            else:
                self._auto_stage = 3

        # reward multipliers (门控 + 缓慢引入)
        # 注意：我们使用事件/增量 shaping，因此这些倍率不会造成“每步刷分”。
        if phase == "approach":
            mult = {
                "distance": 3.0,
                "near_object": 2.0,
                "grasp_guidance": 0.0,
                "single_contact": 0.0,
                "both_contact": 0.0,
                "lift": 0.0,
                "success": 1.0,
            }
        elif phase == "contact":
            mult = {
                "distance": 2.0,
                "near_object": 1.0,
                "grasp_guidance": 0.2,
                "single_contact": 1.0,
                "both_contact": 1.0,
                "lift": 0.0,
                "success": 1.0,
            }
        elif phase == "grasp":
            mult = {
                "distance": 1.0,
                "near_object": 0.5,
                "grasp_guidance": 1.0,
                "single_contact": 1.0,
                "both_contact": 1.2,
                "lift": 0.5,
                "success": 1.0,
            }
        else:  # lift
            mult = {
                "distance": 0.5,
                "near_object": 0.2,
                "grasp_guidance": 0.5,
                "single_contact": 1.0,
                "both_contact": 1.0,
                "lift": 1.0,
                "success": 1.0,
            }

        # apply
        self.reward_weights = self.base_reward_weights.copy()
        for k, m in mult.items():
            self.reward_weights[k] = float(self.base_reward_weights.get(k, 0.0)) * float(m)

        # keep penalties as-is
        for k in ["time_penalty", "drop_penalty", "table_contact_penalty", "out_of_bounds_penalty"]:
            self.reward_weights[k] = float(self.base_reward_weights.get(k, 0.0))

    def get_current_stage(self) -> Dict:
        return {
            "progress": float(self.training_progress),
            "curriculum_phase": str(self.curriculum_phase),
            "difficulty_stage": int(max(self.curriculum_stage, self._auto_stage)),
        }

    def get_reward_weights(self) -> Dict:
        return self.reward_weights.copy()

    # ---------------------------------------------------------------------
    # Observation
    # ---------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # gripper base position
        if self.gripper_base_id >= 0:
            gripper_pos = self.data.xpos[self.gripper_base_id].copy()
            if self.gripper_dof_start is not None:
                gripper_vel = self.data.qvel[self.gripper_dof_start : self.gripper_dof_start + 3].copy()
            else:
                gripper_vel = np.zeros(3)
        else:
            gripper_pos = np.zeros(3)
            gripper_vel = np.zeros(3)

        # driver joint pos/vel
        if self.right_driver_qpos is not None and self.left_driver_qpos is not None:
            driver_pos = np.array([self.data.qpos[self.right_driver_qpos], self.data.qpos[self.left_driver_qpos]], dtype=np.float32)
            right_dof = int(self.model.jnt_dofadr[self.right_driver_joint])
            left_dof = int(self.model.jnt_dofadr[self.left_driver_joint])
            driver_vel = np.array([self.data.qvel[right_dof], self.data.qvel[left_dof]], dtype=np.float32)
        else:
            driver_pos = np.zeros(2, dtype=np.float32)
            driver_vel = np.zeros(2, dtype=np.float32)

        # object pos/vel
        if self.target_body_id >= 0:
            obj_pos = self.data.xpos[self.target_body_id].copy()
            obj_vel_full = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.target_body_id, obj_vel_full, 0)
            obj_vel = obj_vel_full[:3]
        else:
            obj_pos = np.zeros(3)
            obj_vel = np.zeros(3)

        ctrl_normalized = np.array([self.current_gripper_ctrl / 255.0], dtype=np.float32)

        left_contact, right_contact, _, _ = self._check_contacts()
        contact_flag = np.array([1.0 if (left_contact or right_contact) else 0.0], dtype=np.float32)

        obs = np.concatenate(
            [
                gripper_pos,
                gripper_vel,
                driver_pos,
                driver_vel,
                obj_pos,
                obj_vel,
                ctrl_normalized,
                contact_flag,
            ]
        )
        return obs.astype(np.float32)

    # ---------------------------------------------------------------------
    # Contact checks
    # ---------------------------------------------------------------------

    def _check_contacts(self) -> Tuple[bool, bool, bool, bool]:
        """Return (left_contact_target, right_contact_target, left_contact_table, right_contact_table)."""
        left_t = right_t = False
        left_table = right_table = False

        if self.data.ncon <= 0:
            return left_t, right_t, left_table, right_table

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])

            # target contacts
            if self.target_body_id >= 0 and (b1 == self.target_body_id or b2 == self.target_body_id):
                other = b2 if b1 == self.target_body_id else b1
                if other == self.left_pad_body:
                    left_t = True
                elif other == self.right_pad_body:
                    right_t = True

            # table contacts (pad-table)
            if self.table_body_id >= 0 and (b1 == self.table_body_id or b2 == self.table_body_id):
                other = b2 if b1 == self.table_body_id else b1
                if other == self.left_pad_body:
                    left_table = True
                elif other == self.right_pad_body:
                    right_table = True

        return left_t, right_t, left_table, right_table

    # ---------------------------------------------------------------------
    # Reset / Step
    # ---------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random.seed(int(seed))

        mujoco.mj_resetData(self.model, self.data)

        # Determine difficulty stage (manual floor vs auto)
        stage = int(max(self.curriculum_stage, self._auto_stage))

        # Randomize object position (curriculum-controlled)
        if self.target_body_id >= 0 and self.target_qpos_start is not None:
            curriculum_configs = [
                {"xy_range": 0.03, "max_dist": 0.12},
                {"xy_range": 0.06, "max_dist": 0.16},
                {"xy_range": 0.10, "max_dist": 0.20},
                {"xy_range": 0.12, "max_dist": 0.25},
            ]
            stage_cfg = curriculum_configs[int(np.clip(stage, 0, 3))]

            randomize_cfg = self.config.get("randomization", {})
            obj_xy_range = float(randomize_cfg.get("object_xy_range", stage_cfg["xy_range"]))
            obj_z_base = float(randomize_cfg.get("object_z_base", 0.32))
            obj_z_noise = float(randomize_cfg.get("object_z_noise", 0.0))
            max_init_distance = float(randomize_cfg.get("max_init_distance", stage_cfg["max_dist"]))
            max_retries = int(randomize_cfg.get("max_retries", 100))

            valid_position = False
            for _ in range(max_retries):
                random_x = float(self.np_random.uniform(-obj_xy_range, obj_xy_range))
                random_y = float(self.np_random.uniform(-obj_xy_range, obj_xy_range))
                random_z = float(obj_z_base + self.np_random.uniform(-obj_z_noise, obj_z_noise))

                self.data.qpos[self.target_qpos_start + 0] = random_x
                self.data.qpos[self.target_qpos_start + 1] = random_y
                self.data.qpos[self.target_qpos_start + 2] = random_z

                mujoco.mj_forward(self.model, self.data)

                if self.pinch_site_id >= 0:
                    pinch_pos = self.data.site_xpos[self.pinch_site_id]
                    obj_pos = self.data.xpos[self.target_body_id]
                    init_distance = float(np.linalg.norm(pinch_pos - obj_pos))
                    if init_distance <= max_init_distance:
                        valid_position = True
                        break
                else:
                    valid_position = True
                    break

            if not valid_position and max_init_distance < 1.0:
                import warnings

                warnings.warn(
                    f"无法在 {max_retries} 次尝试内采样到满足初始距离约束的位置 "
                    f"(max_init_distance={max_init_distance:.3f}m)。"
                )

        # init gripper (open)
        self.current_gripper_ctrl = 0.0
        self.data.ctrl[0] = self.current_gripper_ctrl

        # (optional) clamp mocap to safe workspace at reset
        if self.gripper_mocap_id >= 0:
            pos = self.data.mocap_pos[self.gripper_mocap_id]
            pos[0] = float(np.clip(pos[0], self.x_min, self.x_max))
            pos[1] = float(np.clip(pos[1], self.y_min, self.y_max))
            pos[2] = float(np.clip(pos[2], self.z_min, self.z_max))

        mujoco.mj_forward(self.model, self.data)

        # warmup steps
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # baseline object height
        if self.target_body_id >= 0:
            self.initial_obj_height = float(self.data.xpos[self.target_body_id][2])
        else:
            self.initial_obj_height = 0.0

        # reset runtime state
        self.steps = 0
        self._render_counter = 0
        self.success_hold = 0
        self._success_ever = False

        self.prev_dist_3d = None
        self.prev_obj_height = None
        self.prev_ctrl = float(self.current_gripper_ctrl)
        self.near_once = False
        self.single_contact_once = False
        self.both_contact_once = False

        obs = self._get_obs()
        info = {
            "curriculum_phase": self.curriculum_phase,
            "difficulty_stage": int(max(self.curriculum_stage, self._auto_stage)),
        }
        return obs, info

    def _clamp_mocap(self):
        if self.gripper_mocap_id < 0:
            return
        pos = self.data.mocap_pos[self.gripper_mocap_id]
        pos[0] = float(np.clip(pos[0], self.x_min, self.x_max))
        pos[1] = float(np.clip(pos[1], self.y_min, self.y_max))
        pos[2] = float(np.clip(pos[2], self.z_min, self.z_max))

    def step(self, action):
        self.steps += 1

        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        if action.shape[0] >= 4:
            dx, dy, dz, gripper_action = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        elif action.shape[0] == 1:
            dx, dy, dz, gripper_action = 0.0, 0.0, 0.0, float(action[0])
        else:
            dx = dy = dz = 0.0
            gripper_action = 0.0

        # --- Gripper control ---
        # ctrl=0 open, ctrl=255 close
        target_ctrl = 127.5 * (1.0 + gripper_action)
        delta = float(np.clip(target_ctrl - self.current_gripper_ctrl, -self.gripper_max_delta, self.gripper_max_delta))
        self.current_gripper_ctrl = float(np.clip(self.current_gripper_ctrl + delta, 0.0, 255.0))
        self.data.ctrl[0] = self.current_gripper_ctrl

        # --- Physics stepping with smooth mocap movement ---
        n_steps = int(self.physics_steps_per_action)
        for _ in range(n_steps):
            if self.gripper_mocap_id >= 0:
                self.data.mocap_pos[self.gripper_mocap_id][0] += (dx * self.position_scale) / n_steps
                self.data.mocap_pos[self.gripper_mocap_id][1] += (dy * self.position_scale) / n_steps
                self.data.mocap_pos[self.gripper_mocap_id][2] += (dz * self.position_scale) / n_steps
                self._clamp_mocap()
            mujoco.mj_step(self.model, self.data)

        # --- state ---
        left_t, right_t, left_table, right_table = self._check_contacts()
        any_contact = left_t or right_t
        both_contact = left_t and right_t

        obj_pos = self.data.xpos[self.target_body_id].copy() if self.target_body_id >= 0 else np.zeros(3)
        obj_height = float(obj_pos[2])
        height_gain = max(0.0, obj_height - float(self.initial_obj_height))

        # gripper position for distance
        if self.pinch_site_id >= 0:
            gripper_pos = self.data.site_xpos[self.pinch_site_id].copy()
        elif self.gripper_base_id >= 0:
            gripper_pos = self.data.xpos[self.gripper_base_id].copy()
        else:
            gripper_pos = np.zeros(3)

        dist_3d = float(np.linalg.norm(gripper_pos - obj_pos))
        dist_xy = float(np.linalg.norm(gripper_pos[:2] - obj_pos[:2]))

        # alignment
        is_aligned = (dist_xy < self.align_xy_thresh) and (abs(float(gripper_pos[2] - obj_pos[2])) < self.align_z_thresh)

        # -----------------------------------------------------------------
        # Reward (final scheme)
        # -----------------------------------------------------------------
        w = self.reward_weights

        # 1) distance shaping (potential-based): only rewards progress
        if self.prev_dist_3d is None:
            dist_rew = 0.0
        else:
            dist_rew = w["distance"] * (np.tanh(self.prev_dist_3d * self.dist_alpha) - np.tanh(dist_3d * self.dist_alpha))
        self.prev_dist_3d = dist_3d

        # 2) near event (once per episode)
        near_rew = 0.0
        if (dist_3d < self.near_threshold) and (not self.near_once):
            near_rew = w["near_object"]
            self.near_once = True

        # 3) grasp guidance: reward closing DELTA when aligned
        ctrl_delta = (float(self.current_gripper_ctrl) - float(self.prev_ctrl)) / 255.0
        grasp_rew = 0.0
        if is_aligned and ctrl_delta > 0.0:
            grasp_rew = w["grasp_guidance"] * ctrl_delta
        self.prev_ctrl = float(self.current_gripper_ctrl)

        # 4) contact events (once per episode)
        contact_rew = 0.0
        if any_contact and (not self.single_contact_once):
            contact_rew += w["single_contact"]
            self.single_contact_once = True
        if both_contact and (not self.both_contact_once):
            contact_rew += w["both_contact"]
            self.both_contact_once = True

        # 5) lift shaping: reward height increment, gated by stable contact
        lift_rew = 0.0
        if self.prev_obj_height is None:
            delta_h = 0.0
        else:
            delta_h = obj_height - float(self.prev_obj_height)
        if both_contact and (delta_h > 0.0):
            lift_rew = w["lift"] * float(delta_h)
        self.prev_obj_height = obj_height

        # 6) penalties
        # time penalty
        penalty = float(w["time_penalty"])

        # pad-table penalty (discourage pushing the table)
        table_pen = 0.0
        if left_table or right_table:
            table_pen = float(w["table_contact_penalty"])
        penalty += table_pen

        # out-of-bounds penalty (mocap is clamped; but if user disables clamp, still safe)
        oob_pen = 0.0
        if self.gripper_mocap_id >= 0:
            mp = self.data.mocap_pos[self.gripper_mocap_id]
            if (mp[0] <= self.x_min + 1e-6) or (mp[0] >= self.x_max - 1e-6) or (mp[1] <= self.y_min + 1e-6) or (mp[1] >= self.y_max - 1e-6):
                oob_pen = float(w["out_of_bounds_penalty"])
                penalty += oob_pen

        # drop penalty: object fell below table by margin OR moved far outside table extents
        drop_pen = 0.0
        # below table top (with margin)
        if obj_height < (self.table_top_z - 0.02):
            drop_pen = float(w["drop_penalty"])
        # far away in XY
        if abs(float(obj_pos[0])) > 0.45 or abs(float(obj_pos[1])) > 0.45:
            drop_pen = min(drop_pen, float(w["drop_penalty"]))
        penalty += drop_pen

        # 7) final success (fixed)
        success_contact = both_contact if self.require_both_contact else any_contact
        final_cond = (height_gain > self.min_lift_height) and bool(success_contact)
        if final_cond:
            self.success_hold += 1
        else:
            self.success_hold = 0

        final_success = bool(self.success_hold >= self.hold_steps)

        success_rew = 0.0
        if final_success and (not self._success_ever):
            success_rew = float(w["success"])
            self._success_ever = True

        # total reward
        reward = float(dist_rew + near_rew + grasp_rew + contact_rew + lift_rew + success_rew + penalty)

        # -----------------------------------------------------------------
        # Termination
        # -----------------------------------------------------------------
        terminated = False
        truncated = bool(self.steps >= self.max_episode_steps)

        # terminate only on FINAL success; curriculum stages do NOT terminate
        if final_success:
            terminated = True

        # If you want to early-terminate on catastrophic drop, enable here.
        # To avoid "early terminate saves time penalty" exploit, we additionally apply remaining time penalty.
        if (drop_pen < 0.0) and (not terminated):
            # terminate on drop to save compute
            terminated = True
            remaining = max(0, self.max_episode_steps - self.steps)
            reward += float(remaining) * float(w["time_penalty"])  # time_penalty is negative

        # bookkeeping
        if terminated or truncated:
            self.success_history.append(1 if final_success else 0)

        obs = self._get_obs()
        info = {
            "is_success": bool(final_success),
            "curriculum_phase": str(self.curriculum_phase),
            "difficulty_stage": int(max(self.curriculum_stage, self._auto_stage)),
            # diagnostics
            "dist_3d": float(dist_3d),
            "height_gain": float(height_gain),
            "left_contact": bool(left_t),
            "right_contact": bool(right_t),
            "left_table_contact": bool(left_table),
            "right_table_contact": bool(right_table),
            "success_hold": int(self.success_hold),
            # reward components (方便 TensorBoard 分解)
            "reward_dist": float(dist_rew),
            "reward_near": float(near_rew),
            "reward_grasp": float(grasp_rew),
            "reward_contact": float(contact_rew),
            "reward_lift": float(lift_rew),
            "reward_success": float(success_rew),
            "reward_table": float(table_pen),
            "reward_oob": float(oob_pen),
            "reward_drop": float(drop_pen),
            "reward_penalty": float(penalty),
        }

        self._render_counter += 1
        if self.render_mode != "None" and (self._render_counter % max(1, self.render_every) == 0):
            self._render()

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    def _render(self):
        if self.render_mode == "None":
            return
        try:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.sync()
        except Exception:
            pass

    def render(self):
        if self.render_mode == "human":
            self._render()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=-1)
            return self._renderer.render()
        return None

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        self._renderer = None


if __name__ == "__main__":
    # 简单 sanity check（不依赖训练脚本）
    os.environ.setdefault("MUJOCO_GL", "glfw")

    cfg = {
        "env": {"max_episode_steps": 200, "physics_steps_per_action": 10},
        "control": {"position_scale": 0.01, "gripper_max_delta": 20.0, "z_min": 0.33, "z_max": 0.60},
        "reward_weights": {
            "distance": 1.0,
            "near_object": 2.0,
            "grasp_guidance": 1.0,
            "single_contact": 5.0,
            "both_contact": 20.0,
            "lift": 300.0,
            "success": 200.0,
            "time_penalty": -0.01,
            "drop_penalty": -50.0,
            "table_contact_penalty": -0.5,
        },
        "success_criteria": {"min_lift_height": 0.02, "require_both_contact": True, "hold_steps": 10},
        "curriculum": {"enabled": True},
    }

    env = Robotiq2F85Env(render_mode="human", config=cfg)
    obs, info = env.reset()
    print("obs shape:", obs.shape, "info:", info)

    for t in range(200):
        # random actions
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if (t % 20) == 0:
            print(f"t={t:03d} r={r:+.3f} dist={info['dist_3d']:.3f} hg={info['height_gain']:.3f} succ_hold={info['success_hold']}")
        if term or trunc:
            print("done", term, trunc, "is_success", info.get("is_success"))
            obs, info = env.reset()

    env.close()