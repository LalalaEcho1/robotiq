#!/usr/bin/env python3
"""
Robotiq 2F-85 夹爪抓取测试脚本
用于验证夹爪是否能正常移动和抓取物体

测试流程:
1. 移动到物体上方
2. 智能下降 (基于pad高度与物体高度对比)
3. 闭合夹爪
4. 抬起物体
5. 验证抓取成功
"""

import numpy as np
import time
import os
import mujoco

os.environ["MUJOCO_GL"] = "glfw"

from envs.robotiq_env import Robotiq2F85Env


def test_robotiq_grasp():
    """测试 Robotiq 2F-85 夹爪抓取功能"""
    print("=" * 60)
    print("🔧 Robotiq 2F-85 夹爪抓取测试 (智能下降)")
    print("=" * 60)
    
    env = Robotiq2F85Env(render_mode='human', max_episode_steps=400, object_id=12)
    env.set_training_progress(1.0)  # 使用完整的lift成功标准
    obs, _ = env.reset()
    
    # 获取关键位置ID
    right_pad_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "right_pad")
    left_pad_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "left_pad")
    pinch_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    
    # 解析初始观测
    gripper_base_pos = obs[:3]  # 夹爪基座位置
    obj_pos = obs[10:13]        # 物体位置
    
    # 获取Pinch Site位置（与奖励函数一致）
    pinch_pos = env.data.site_xpos[pinch_site_id]
    obj_actual_pos = env.data.xpos[env.target_body_id]
    
    # 获取实际pad高度
    pad_z = (env.data.xpos[right_pad_id][2] + env.data.xpos[left_pad_id][2]) / 2
    
    # 计算真实距离（Pinch Site到物体）
    init_dist_3d = np.linalg.norm(pinch_pos - obj_actual_pos)
    init_dist_xy = np.linalg.norm(pinch_pos[:2] - obj_actual_pos[:2])
    
    print(f"\n📊 初始状态:")
    print(f"  夹爪基座位置: ({gripper_base_pos[0]:.3f}, {gripper_base_pos[1]:.3f}, {gripper_base_pos[2]:.3f})")
    print(f"  Pinch Site位置: ({pinch_pos[0]:.3f}, {pinch_pos[1]:.3f}, {pinch_pos[2]:.4f})")
    print(f"  Finger Pad高度: {pad_z:.4f} m")
    print(f"  物体位置: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
    print(f"  真实XY距离 (Pinch→物体): {init_dist_xy*100:.2f} cm")
    print(f"  真实3D距离 (Pinch→物体): {init_dist_3d*100:.2f} cm")
    
    obj_start_z = obj_pos[2]
    total_reward = 0
    
    # ==================== 阶段 1: 水平对准 ====================
    print("\n⬅️➡️ 阶段 1: 水平对准物体...")
    for step in range(50):
        obj_pos = obs[10:13]
        dx_error = obj_pos[0] - obs[0]
        dy_error = obj_pos[1] - obs[1]
        
        dx = np.clip(dx_error * 5.0, -1, 1)
        dy = np.clip(dy_error * 5.0, -1, 1)
        
        action = np.array([dx, dy, 0.0, -1.0])
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if step % 15 == 0:
            # 使用Pinch Site计算真实距离
            pinch_pos = env.data.site_xpos[pinch_site_id]
            obj_pos_actual = env.data.xpos[env.target_body_id]
            dist_xy = np.linalg.norm(pinch_pos[:2] - obj_pos_actual[:2])
            print(f"  Step {step}: XY距离 (Pinch→物体) = {dist_xy*100:.2f} cm")
        
        time.sleep(0.01)
    
    pinch_pos = env.data.site_xpos[pinch_site_id]
    obj_pos_actual = env.data.xpos[env.target_body_id]
    dist_xy_final = np.linalg.norm(pinch_pos[:2] - obj_pos_actual[:2])
    print(f"  对准完成: XY距离 (Pinch→物体) = {dist_xy_final*100:.2f} cm")
    
    # ==================== 阶段 2: 智能下降 ====================
    print("\n⬇️ 阶段 2: 智能下降到抓取位置...")
    # 物体高度 + 一点余量 (pad需要在物体中心高度)
    target_pad_z = obs[12] + 0.005  # 物体高度 + 5mm
    
    for step in range(100):
        obj_pos = obs[10:13]
        
        # 获取当前pad高度
        pad_z = (env.data.xpos[right_pad_id][2] + env.data.xpos[left_pad_id][2]) / 2
        
        # 继续水平对准
        dx_error = obj_pos[0] - obs[0]
        dy_error = obj_pos[1] - obs[1]
        dx = np.clip(dx_error * 3.0, -0.3, 0.3)
        dy = np.clip(dy_error * 3.0, -0.3, 0.3)
        
        # 智能下降控制 - 基于pad高度
        height_diff = pad_z - target_pad_z
        if height_diff > 0.01:  # pad还在物体上方 > 1cm
            dz = np.clip(-height_diff * 3.0, -0.5, 0.0)  # 缓慢下降
        else:
            dz = 0.0  # 到达目标高度，停止下降
        
        action = np.array([dx, dy, dz, -1.0])
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"  Step {step}: pad_z={pad_z:.4f} m, obj_z={obs[12]:.4f} m, diff={height_diff:.4f} m")
        
        # 如果已经到达目标高度，提前结束
        if abs(height_diff) < 0.01:
            print(f"  到达目标高度! pad_z={pad_z:.4f} m")
            break
        
        time.sleep(0.01)
    
    print(f"  下降完成: pad高度={pad_z:.4f} m, 物体高度={obs[12]:.4f} m")
    
    # ==================== 阶段 3: 闭合夹爪 ====================
    print("\n👊 阶段 3: 闭合夹爪...")
    for step in range(50):
        action = np.array([0.0, 0.0, 0.0, 1.0])  # 闭合夹爪，不移动
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if step % 15 == 0:
            print(f"  Step {step}: ctrl={info['gripper_ctrl']:.1f}, "
                  f"接触=L:{info['left_contact']}/R:{info['right_contact']}")
        
        time.sleep(0.01)
    
    print(f"  闭合完成: ctrl={info['gripper_ctrl']:.1f}")
    print(f"  接触状态: 左={info['left_contact']}, 右={info['right_contact']}")
    
    # ==================== 阶段 4: 抬起物体 ====================
    print("\n⬆️ 阶段 4: 抬起物体...")
    for step in range(80):
        action = np.array([0.0, 0.0, 0.5, 1.0])  # 向上 + 保持闭合
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            lift = (obs[12] - obj_start_z) * 100
            print(f"  Step {step}: 物体高度={obs[12]:.4f} m, "
                  f"抬升={lift:.1f} cm, 接触=L:{info['left_contact']}/R:{info['right_contact']}")
        
        if term:
            print(f"  🎉 触发成功条件!")
            break
        
        time.sleep(0.01)
    
    # ==================== 结果统计 ====================
    obj_final_z = obs[12]
    total_lift = (obj_final_z - obj_start_z) * 100
    
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print(f"  初始物体高度: {obj_start_z:.4f} m")
    print(f"  最终物体高度: {obj_final_z:.4f} m")
    print(f"  抬升高度: {total_lift:.1f} cm")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  成功标志: {info['is_success']}")
    
    if total_lift > 2:
        print("\n✅ 抓取成功! 夹爪能正常移动和抓取物体。")
        success = True
    else:
        print("\n❌ 抓取失败，需要调整参数。")
        success = False
    
    print("=" * 60)
    
    # 保持显示
    print("\n保持显示 3 秒...")
    for _ in range(150):
        action = np.array([0.0, 0.0, 0.0, 1.0])
        env.step(action)
        time.sleep(0.02)
    
    env.close()
    return success


def test_robotiq_simple():
    """简单测试 - 无可视化"""
    print("=" * 60)
    print("🔧 Robotiq 简单功能测试 (无可视化)")
    print("=" * 60)
    
    env = Robotiq2F85Env(render_mode='None', max_episode_steps=200)
    
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    obs, _ = env.reset()
    print(f"初始观测形状: {obs.shape}")
    print(f"夹爪位置: ({obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f})")
    print(f"物体位置: ({obs[10]:.3f}, {obs[11]:.3f}, {obs[12]:.3f})")
    
    # 测试移动
    print("\n测试移动...")
    for _ in range(20):
        action = np.array([0.5, 0.0, -0.5, 0.0])
        obs, _, _, _, _ = env.step(action)
    
    print(f"移动后夹爪位置: ({obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f})")
    
    # 测试夹爪
    print("\n测试夹爪闭合...")
    for _ in range(20):
        action = np.array([0.0, 0.0, 0.0, 1.0])
        obs, _, _, _, info = env.step(action)
    
    print(f"夹爪控制值: {info['gripper_ctrl']:.1f}")
    
    env.close()
    print("\n✅ 简单测试完成!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robotiq 2F-85 夹爪测试")
    parser.add_argument("--simple", action="store_true", help="运行简单测试 (无可视化)")
    args = parser.parse_args()
    
    if args.simple:
        test_robotiq_simple()
    else:
        success = test_robotiq_grasp()
        exit(0 if success else 1)
