#!/usr/bin/env python3
import numpy as np
import math
# import rclpy # Removed as it's not used directly in the planner logic here
# from rclpy.node import Node # Removed
import yaml
import os
import tkinter as tk
from tkinter import ttk

class CollisionChecker:
    """一个包含碰撞检测算法的静态工具类"""

    @staticmethod
    def _get_rotated_rect_vertices(cx, cy, length, width, theta):
        """计算旋转矩形的四个顶点坐标"""
        hl, hw = length / 2.0, width / 2.0
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        
        corners = [
            (-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)
        ]
        
        return [
            (
                cx + c[0] * cos_theta - c[1] * sin_theta,
                cy + c[0] * sin_theta + c[1] * cos_theta
            ) for c in corners
        ]

    @staticmethod
    def _project_polygon(vertices, axis):
        """将多边形的顶点投影到轴上"""
        dots = [v[0] * axis[0] + v[1] * axis[1] for v in vertices]
        return min(dots), max(dots)

    @staticmethod
    def _get_axes(vertices):
        """获取多边形的所有投影轴（边的法向量）"""
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # 获得法向量
            normal = (-edge[1], edge[0])
            axes.append(normal)
        return axes

    @staticmethod
    def check_rect_rect_collision(rect1_verts, rect2_verts):
        """使用SAT检查两个矩形是否碰撞"""
        axes = CollisionChecker._get_axes(rect1_verts) + CollisionChecker._get_axes(rect2_verts)

        for axis in axes:
            # 归一化轴向量
            mag = math.sqrt(axis[0]**2 + axis[1]**2)
            if mag == 0: continue
            axis_norm = (axis[0] / mag, axis[1] / mag)

            min1, max1 = CollisionChecker._project_polygon(rect1_verts, axis_norm)
            min2, max2 = CollisionChecker._project_polygon(rect2_verts, axis_norm)

            if max1 < min2 or max2 < min1:
                return False  # 找到分离轴，没有碰撞
        
        return True # 所有轴上都重叠，存在碰撞

    @staticmethod
    def check_rect_circle_collision(rect_verts, cx, cy, radius):
        """使用SAT检查矩形和圆形是否碰撞"""
        # 1. 检查矩形的法线轴
        axes = CollisionChecker._get_axes(rect_verts)
        for axis in axes:
            mag = math.sqrt(axis[0]**2 + axis[1]**2)
            if mag == 0: continue
            axis_norm = (axis[0] / mag, axis[1] / mag)
            
            min_r, max_r = CollisionChecker._project_polygon(rect_verts, axis_norm)
            
            # 投影圆心
            c_proj = (cx * axis_norm[0] + cy * axis_norm[1])
            min_c, max_c = c_proj - radius, c_proj + radius

            if max_r < min_c or max_c < min_r:
                return False

        # 2. 检查从圆心到矩形最近顶点的轴
        closest_vertex = min(rect_verts, key=lambda v: (v[0] - cx)**2 + (v[1] - cy)**2)
        axis = (closest_vertex[0] - cx, closest_vertex[1] - cy)
        mag = math.sqrt(axis[0]**2 + axis[1]**2)
        if mag == 0: return True # 圆心在顶点上，碰撞
        axis_norm = (axis[0] / mag, axis[1] / mag)

        min_r, max_r = CollisionChecker._project_polygon(rect_verts, axis_norm)
        c_proj = (cx * axis_norm[0] + cy * axis_norm[1])
        min_c, max_c = c_proj - radius, c_proj + radius

        if max_r < min_c or max_c < min_r:
            return False

        return True

class Config:
    # 机器人参数配置：最大速度、加速度、角速度等
    # This class is defined but not used in the provided snippet.
    # If it's meant to be used, DWAPlanner should perhaps take an instance of it.
    def __init__(self):
        pass

class DWAPlanner:
    """动态窗口法(Dynamic Window Approach)路径规划器"""
    
    def __init__(self, config=None, config_manager=None):
        # 获取配置
        if config_manager:
            default_config = config_manager.get_config('dwa')
        else:
            # DWA参数 - 这些将作为基础配置
            default_config = {
                'max_speed': 5.0,         # 最大线速度 (m/s)
                'min_speed': 0.0,         # 最小线速度 (m/s) (可以是负数，如果允许倒退)
                'max_omega': 0.25,         # 最大角速度 (rad/s)
                'min_omega': -0.25,        # 最小角速度 (rad/s)
                'max_accel': 0.05,         # 最大线加速度 (m/s^2)
                'max_domega': 0.05,        # 最大角加速度 (rad/s^2)
                'v_resolution': 0.1,      # 线速度采样分辨率 (m/s)
                'omega_resolution': 0.02,  # 角速度采样分辨率 (rad/s)
                'dt': 0.1,                # 控制命令的时间步长 (s)
                'predict_time': 8.0,      # 轨迹预测时间 (s)
                'base_heading_weight': 1.2,    # 基础航向权重
                'base_dist_weight': 0.6,       # 基础目标距离权重
                'base_velocity_weight': 0.5,   # 基础速度权重
                'base_obstacle_weight': 2.5,   # 大幅提高障碍物权重，使其更谨慎
                'base_safe_distance': 20.0,     # 增加安全距离，为船体机动预留更多空间
                'robot_length': 69.80,         # 机器人/船体长度 (m) - 根据实际船舶参数更新
                'robot_width': 13.44,          # 机器人/船体宽度 (m) - 根据实际船舶参数更新
                'num_body_circles': 5,         # 用于碰撞检测的船体包围圆数量
                'control_point_offset': 30.0,  # 控制点前向偏移量 (m)，从几何中心沿船头方向
            }
        
        # 使用提供的配置或默认配置
        self.base_config = default_config.copy()
        if config:
            self.base_config.update(config)
    
    def dwa_planning(self, current_pos, current_vel, current_omega, goal, obstacles, current_state="NORMAL_NAVIGATION"):
        """
        使用动态窗口法进行局部路径规划和避障
        
        参数:
        current_pos -- 当前位置和朝向 [x, y, theta] (theta单位：弧度)
        current_vel -- 当前线速度 (m/s)
        current_omega -- 当前角速度 (rad/s)
        goal -- 目标位置 [x, y] (通常是全局路径上的前瞻点)
        obstacles -- 障碍物列表 [[x1, y1, r1], [x2, y2, r2], ...] (r是障碍物半径)
        current_state -- 当前的行为状态 (str)，由外部状态机提供，用于动态调整参数
        
        返回:
        best_v -- 最优线速度 (m/s)
        best_omega -- 最优角速度 (rad/s)
        best_trajectory -- 最优轨迹点列表 [[x,y], ...] 或 None (如果无有效路径)
        """
        
        # --- 0. 参数准备 ---
        # 从基础配置中获取机器人性能参数
        max_speed = self.base_config['max_speed']
        min_speed = self.base_config['min_speed']
        max_omega = self.base_config['max_omega']
        min_omega = self.base_config['min_omega']
        max_accel = self.base_config['max_accel']
        max_domega = self.base_config['max_domega']
        v_resolution = self.base_config['v_resolution']
        omega_resolution = self.base_config['omega_resolution']
        dt = self.base_config['dt']
        predict_time = self.base_config['predict_time']
        
        # 加载基础权重和安全距离
        active_heading_weight = self.base_config['base_heading_weight']
        active_dist_weight = self.base_config['base_dist_weight']
        active_velocity_weight = self.base_config['base_velocity_weight']
        active_obstacle_weight = self.base_config['base_obstacle_weight']
        active_safe_distance = self.base_config['base_safe_distance']
        
        # --- 新增: "启动增益"逻辑 ---
        # 如果船几乎是静止的，就进入一个临时的"启动模式"，强制让前进的奖励远大于避障惩罚
        STARTUP_SPEED_THRESHOLD = 0.2  # (m/s) 当速度低于此阈值时，视为启动阶段
        if abs(current_vel) < STARTUP_SPEED_THRESHOLD:
            # print("DEBUG: Applying STARTUP BOOST") # 用于调试
            active_velocity_weight *= 2.0   # 放大前进奖励
            active_obstacle_weight *= 0.2   # 大幅降低障碍物惩罚，变得极其勇敢
            active_safe_distance *= 0.5     # 临时将安全距离减半
            active_heading_weight *= 1.5    # 核心修正: 启动时更要坚定地朝向目标！
        
        # 机动偏好，例如："PREFER_RIGHT_TURN", "PREFER_LEFT_TURN", "MAINTAIN_COURSE"
        maneuver_preference = None
        # Omega惩罚/奖励的强度因子，可以根据状态调整
        maneuver_factor = active_obstacle_weight * 1.5 # 使其与避障权重有一定关联

        # --- 1. 根据当前状态 (current_state) 动态调整参数 ---
        if current_state == "HEAD_ON_RIGHT_TURN": # 对遇右转
            active_obstacle_weight *= 2.0
            active_heading_weight *= 0.3 # 降低对原始航向的执着，为避让留空间
            active_velocity_weight *= 0.7 # 可能需要减速
            active_safe_distance *= 1.2
            maneuver_preference = "PREFER_RIGHT_TURN"
            # print(f"INFO: DWA State: HEAD_ON_RIGHT_TURN")
        elif current_state == "CROSSING_GIVE_WAY_RIGHT": # 交叉让路（右转从船尾过）
            active_obstacle_weight *= 2.5
            active_heading_weight *= 0.2
            active_velocity_weight *= 0.5 # 通常需要显著减速或大幅度转向
            active_safe_distance *= 1.3
            maneuver_preference = "PREFER_RIGHT_TURN" # 也可以是更复杂的避让点
            # print(f"INFO: DWA State: CROSSING_GIVE_WAY_RIGHT")
        elif current_state == "OVERTAKE_MANEUVER_LEFT": # 追越（从左侧）
            active_velocity_weight *= 1.1 # 保持或略微增加速度
            active_obstacle_weight *= 1.5 # 保证横向安全
            maneuver_preference = "PREFER_LEFT_TURN" # 追越时通常选择一侧
            # print(f"INFO: DWA State: OVERTAKE_MANEUVER_LEFT")
        elif current_state == "OVERTAKE_MANEUVER_RIGHT": # 追越（从右侧）
            active_velocity_weight *= 1.1
            active_obstacle_weight *= 1.5
            maneuver_preference = "PREFER_RIGHT_TURN"
            # print(f"INFO: DWA State: OVERTAKE_MANEUVER_RIGHT")
        elif current_state == "EMERGENCY_AVOIDANCE_RIGHT": # 紧急右转避碰
            active_obstacle_weight *= 3.0
            active_heading_weight = 0.01 # 航向几乎不重要，活下来！
            active_velocity_weight = 0.1 # 保持一个微小的速度倾向，避免完全停止
            active_safe_distance *= 1.5
            maneuver_preference = "PREFER_RIGHT_TURN"
            # print(f"INFO: DWA State: EMERGENCY_AVOIDANCE_RIGHT")
        elif current_state == "EMERGENCY_STOP":
            active_velocity_weight = 0.0
            active_obstacle_weight *= 2.0 # 即使停下也要评估周围
            # print(f"INFO: DWA State: EMERGENCY_STOP, returning (0,0)")
            return 0.0, 0.0, [current_pos[:2]] # 立即停止
        elif current_state == "NORMAL_NAVIGATION":
            # print(f"INFO: DWA State: NORMAL_NAVIGATION")
            pass # 使用基础参数
        # --- 可以添加更多状态判断和对应的参数调整逻辑 ---

        # --- 2. 计算动态窗口 ---
        # 速度限制 Vs = [min_speed, max_speed, min_omega, max_omega]
        # 可达速度 Vd
        vd = [
            max(min_speed, current_vel - max_accel * dt),
            min(max_speed, current_vel + max_accel * dt),
            max(min_omega, current_omega - max_domega * dt),
            min(max_omega, current_omega + max_domega * dt)
        ]
        # 最终的动态窗口 dw
        dw = [
            max(min_speed, vd[0]), # 确保不低于全局最小速度
            min(max_speed, vd[1]), # 确保不高于全局最大速度
            max(min_omega, vd[2]),
            min(max_omega, vd[3])
        ]

        # --- 3. 轨迹采样与评估 ---
        best_score = -float('inf')
        best_v = 0.0
        best_omega = 0.0
        best_trajectory = None
        
        # 必须提供当前朝向 theta
        if len(current_pos) < 3:
            # print("ERROR: DWAPlanner: current_pos must include theta [x, y, theta]. Assuming theta=0.")
            # 这应该由调用者保证，或者在此处抛出异常
            current_theta = 0.0 
        else:
            current_theta = current_pos[2]
        
        # --- 新增: 计算控制点 (通常是船头附近) 的起始位置 ---
        control_point_offset = self.base_config.get('control_point_offset', 0.0)
        current_x, current_y = current_pos[0], current_pos[1]
        
        if control_point_offset != 0.0:
            # 从几何中心向前偏移，得到控制点
            cp_x = current_x + control_point_offset * math.cos(current_theta)
            cp_y = current_y + control_point_offset * math.sin(current_theta)
        else:
            # 如果不偏移，控制点就是几何中心
            cp_x, cp_y = current_x, current_y
        
        num_v_samples = 0
        num_omega_samples = 0

        # 遍历动态窗口中的所有速度对 (v, omega)，核心计算逻辑
        v_idx = 0
        while True:
            v = dw[0] + v_idx * v_resolution
            if v > dw[1]:
                break
            num_v_samples +=1
            
            omega_idx = 0
            while True:
                omega = dw[2] + omega_idx * omega_resolution
                if omega > dw[3]:
                    break
                if v_idx == 0 and omega_idx ==0 : num_omega_samples +=1 # Count only once for all v

                # --- 3a. 预测轨迹 ---
                trajectory = self.predict_trajectory(
                    cp_x, cp_y, current_theta, # 从控制点开始预测
                    v, omega, predict_time, dt
                )
                
                # --- 3b. 评估轨迹 ---
                # 1. 障碍物代价 (cost: 0=safe, 1=close, inf=collision)
                obstacle_cost = self.calc_obstacle_cost(trajectory, obstacles, active_safe_distance, dt)
                
                if obstacle_cost == float('inf'): # 碰撞，此轨迹无效
                    omega_idx += 1
                    continue
                
                # 2. 航向得分 (越高越好)
                # 目标：使机器人朝向目标点 (goal)
                # 这里我们使用轨迹末端点到目标点的方向与当前机器人朝向的差异
                final_pos_in_traj = trajectory[-1]
                # 航向指向目标
                angle_to_goal = math.atan2(goal[1] - final_pos_in_traj[1], goal[0] - final_pos_in_traj[0])
                # 轨迹末端的预测航向
                predicted_final_theta = self.normalize_angle(current_theta + omega * predict_time)
                
                heading_diff = abs(self.normalize_angle(angle_to_goal - predicted_final_theta))
                # (math.pi - heading_diff) / math.pi : 范围 0 (180度差异) 到 1 (0度差异)
                heading_score = (math.pi - heading_diff) / math.pi
                
                # 3. 目标距离得分 (越高越好)
                # 衡量轨迹末端点与目标点的距离，越近越好
                goal_dist = math.sqrt((goal[0] - final_pos_in_traj[0])**2 + (goal[1] - final_pos_in_traj[1])**2)
                # 使用 1 / (1 + dist) 形式，避免除以零，且距离越小得分越高
                dist_score = 1.0 / (1.0 + goal_dist) 
                
                # 4. 速度得分 (越高越好，鼓励前进)
                # 对于船舶，通常不希望负速（倒退），除非特定状态允许
                # 如果 min_speed 可以为负，这里的评价需要调整
                velocity_score = (abs(v) / max_speed) if max_speed > 0 else 0.0
                if v < 0 and current_state != "EMERGENCY_REVERSE": # 惩罚不必要的倒退
                    velocity_score *= 0.1 

                # --- 3c. 综合评分 ---
                # 障碍物得分: (1 - obstacle_cost) 范围 0 (接近危险) 到 1 (安全)
                score = (active_heading_weight * heading_score +
                         active_dist_weight * dist_score +
                         active_velocity_weight * velocity_score +
                         active_obstacle_weight * (1.0 - obstacle_cost)) # obstacle_cost是0-1的值，1-cost则越大越好
                
                # --- 3d. 根据机动偏好调整得分 ---
                # **重要**: 确认 omega 的正负与左/右转的对应关系!
                # 假设: omega < 0 为右转 (顺时针), omega > 0 为左转 (逆时针)
                if maneuver_preference == "PREFER_RIGHT_TURN":
                    if omega > 0.01:  # 如果是左转 (不期望)
                        score -= abs(omega) * maneuver_factor # 惩罚
                    elif omega < -0.01: # 如果是右转 (期望)
                        score += abs(omega) * maneuver_factor * 0.5 # 奖励 (奖励幅度可以小一些)
                elif maneuver_preference == "PREFER_LEFT_TURN":
                    if omega < -0.01: # 如果是右转 (不期望)
                        score -= abs(omega) * maneuver_factor
                    elif omega > 0.01: # 如果是左转 (期望)
                        score += abs(omega) * maneuver_factor * 0.5
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_omega = omega
                    best_trajectory = trajectory
                
                omega_idx += 1
            v_idx += 1
        
        # print(f"DEBUG: DWA samples: v_samples={num_v_samples}, omega_samples_per_v={num_omega_samples}")
        # print(f"DEBUG: DWA best_v={best_v:.2f}, best_omega={best_omega:.2f}, best_score={best_score:.2f}, state={current_state}")

        # --- 最终覆盖机制: 防失速与强制启动模块 ---
        # 如果船几乎静止，并且DWA算法给出的最优解仍然是几乎不前进，
        # 这意味着它陷入了局部最优陷阱（原地转圈）。我们将强行覆盖其决策。
        STARTUP_SPEED_THRESHOLD = 0.2
        if abs(current_vel) < STARTUP_SPEED_THRESHOLD and best_v < 0.01:
            # print("DEBUG: DWA failed to find a forward path. Engaging ANTI-STALL OVERRIDE.")
            
            # 目标: 找到一个最安全的、能够向前移动的轨迹，忽略所有其他评分。
            safest_forward_cost = float('inf')
            override_v, override_omega, override_trajectory = 0.0, 0.0, None

            # 仅在动态窗口内搜索前进的速度
            v_candidates = []
            # 优先尝试动态窗口中的速度
            v_candidates.extend([v for v in np.arange(dw[0], dw[1]+1e-6, v_resolution) if v > 0.0])
            # 如果动态窗口里没有任何正速度(常发生在 max_accel 很小的情况下)，
            # 则人为添加一些极小的正速度候选，以打破僵局。
            if not v_candidates:
                v_candidates = np.linspace(0.01, min(0.3, self.base_config.get('max_speed', 1.0)), 5) # 5 个候选速度

            for v_o in v_candidates:
                # 只测试近似直行的轨迹，避免复杂计算
                for omega_o in np.arange(-0.05, 0.05, omega_resolution):
                    trajectory = self.predict_trajectory(
                        cp_x, cp_y, current_theta, # 防失速也从控制点开始预测
                        v_o, omega_o, predict_time, dt
                    )
                    obstacle_cost = self.calc_obstacle_cost(trajectory, obstacles, active_safe_distance, dt)
                    
                    # 寻找障碍物代价值最低（最安全）的轨迹
                    if obstacle_cost < safest_forward_cost:
                        safest_forward_cost = obstacle_cost
                        override_v = v_o
                        override_omega = omega_o
                        override_trajectory = trajectory
            
            # 如果找到了任何一个不直接碰撞的前进方案，就采用它
            if safest_forward_cost != float('inf'):
                best_v, best_omega, best_trajectory = override_v, override_omega, override_trajectory
                # print(f"DEBUG: OVERRIDE active: v={best_v:.2f}, omega={best_omega:.2f}")

        if best_trajectory is None:
            # print(f"WARN: DWAPlanner: No valid trajectory found for state '{current_state}'. Obstacles: {len(obstacles) if obstacles else 0}.")
            # 如果没有找到有效路径（通常因为所有路径都撞到障碍物或动态窗口太小）
            # 返回0速度，让上层决策或安全层处理
            return 0.0, 0.0, [current_pos[:2]] # 保持当前位置（或最近的安全位置）
            
        return best_v, best_omega, best_trajectory

    def predict_trajectory(self, x, y, theta, v, omega, predict_time, dt):
        """
        预测给定控制下的轨迹 (基于简单匀速圆周运动模型)
        """
        trajectory = []
        current_x, current_y, current_theta = x, y, theta
        
        num_steps = int(predict_time / dt)
        
        for _ in range(num_steps):
            # 对于非常小的omega，近似为直线运动以避免除以零或数值问题
            if abs(omega) < 1e-5: # 阈值，根据需要调整
                current_x += v * math.cos(current_theta) * dt
                current_y += v * math.sin(current_theta) * dt
                # current_theta 保持不变 (因为omega近似为0)
            else:
                # 精确圆弧模型
                radius = v / omega
                delta_theta = omega * dt
                
                # 计算圆心 (cx, cy)
                # cx = current_x - radius * math.sin(current_theta)
                # cy = current_y + radius * math.cos(current_theta)
                # 
                # current_x = cx + radius * math.sin(current_theta + delta_theta)
                # current_y = cy - radius * math.cos(current_theta + delta_theta)
                # current_theta = self.normalize_angle(current_theta + delta_theta)

                # 或者更简单的离散更新（在小dt下通常足够）
                current_x += v * math.cos(current_theta) * dt
                current_y += v * math.sin(current_theta) * dt
                current_theta = self.normalize_angle(current_theta + omega * dt)

            trajectory.append([current_x, current_y, current_theta])
        
        return trajectory

    def _check_hard_collision_sat(self, trajectory, obstacles, dt):
        """
        使用分离轴定理(SAT)检查轨迹是否存在精确的物理碰撞。
        返回 True 如果有碰撞，否则返回 False。
        """
        robot_length = self.base_config.get('robot_length', 0.1)
        robot_width = self.base_config.get('robot_width', 0.1)
        control_point_offset = self.base_config.get('control_point_offset', 0.0)

        for i, traj_point in enumerate(trajectory):
            # traj_point 是控制点的位置和姿态
            cp_x, cp_y, robot_theta = traj_point
            time_at_point = (i + 1) * dt
            
            # 从控制点位置反推几何中心位置
            robot_x = cp_x - control_point_offset * math.cos(robot_theta)
            robot_y = cp_y - control_point_offset * math.sin(robot_theta)

            # 获取当前轨迹点的机器人精确顶点
            robot_verts = CollisionChecker._get_rotated_rect_vertices(
                robot_x, robot_y, robot_length, robot_width, robot_theta
            )

            for obs in obstacles:
                obs_initial_pos = obs.get('position', {})
                obs_vel = obs.get('velocity')
                obs_geometry = obs.get('geometry', {})
                
                obs_x_initial = obs_initial_pos.get('x', 0)
                obs_y_initial = obs_initial_pos.get('y', 0)

                if obs_vel:
                    obs_x_future = obs_x_initial + obs_vel.get('vx', 0) * time_at_point
                    obs_y_future = obs_y_initial + obs_vel.get('vy', 0) * time_at_point
                else:
                    obs_x_future = obs_x_initial
                    obs_y_future = obs_y_initial
                
                # --- SAT 精确碰撞检测 ---
                if obs_geometry.get('type') == 'rectangle':
                    obs_len = obs_geometry.get('length', 1.0)
                    obs_w = obs_geometry.get('width', 1.0)
                    # 从速度计算障碍物朝向
                    obs_vx = obs_vel.get('vx', 0) if obs_vel else 0
                    obs_vy = obs_vel.get('vy', 0) if obs_vel else 0
                    obs_theta = math.atan2(obs_vy, obs_vx)
                    
                    obs_verts = CollisionChecker._get_rotated_rect_vertices(
                        obs_x_future, obs_y_future, obs_len, obs_w, obs_theta
                    )
                    
                    if CollisionChecker.check_rect_rect_collision(robot_verts, obs_verts):
                        return True # 发生碰撞

                elif obs_geometry.get('type') == 'circle':
                    obs_r = obs_geometry.get('radius', 1.0)
                    if CollisionChecker.check_rect_circle_collision(
                        robot_verts, obs_x_future, obs_y_future, obs_r
                    ):
                        return True # 发生碰撞
        
        return False # 轨迹全程无硬碰撞

    def calc_obstacle_cost(self, trajectory, obstacles, current_safe_distance, dt):
        """
        混合策略成本计算:
        1. 使用SAT进行精确的硬碰撞检测。
        2. 如果没有硬碰撞，使用多圆模型计算软碰撞(安全距离)代价。
        """
        if not obstacles:
            return 0.0

        # --- 1. 硬碰撞检测 ---
        if self._check_hard_collision_sat(trajectory, obstacles, dt):
            return float('inf') # 发生物理碰撞，此轨迹无效

        # --- 2. 软碰撞代价计算 (基于多圆模型) ---
        # 如果代码执行到这里，说明没有发生硬碰撞。
        # 我们现在计算一个"靠近"障碍物的代价。
        
        # (这部分代码与我们上一步实现的 "方案一" 逻辑完全相同)
        robot_length = self.base_config.get('robot_length', 0.1)
        num_body_circles = self.base_config.get('num_body_circles', 1)
        control_point_offset = self.base_config.get('control_point_offset', 0.0)
        
        if num_body_circles <= 1:
            body_circle_offsets = [0.0]
            body_circle_radius = robot_length / 2.0
        else:
            spacing = robot_length / num_body_circles
            body_circle_offsets = [(i - (num_body_circles - 1) / 2.0) * spacing for i in range(num_body_circles)]
            body_circle_radius = max(spacing / 2.0, self.base_config.get('robot_width', 0.1) / 2.0)

        min_dist_to_any_obstacle_surface = float('inf')

        for i, traj_point in enumerate(trajectory):
            # traj_point 是控制点的位置和姿态
            cp_x, cp_y, robot_theta = traj_point
            time_at_point = (i + 1) * dt

            # 从控制点位置反推几何中心位置
            robot_x = cp_x - control_point_offset * math.cos(robot_theta)
            robot_y = cp_y - control_point_offset * math.sin(robot_theta)

            for obs in obstacles:
                obs_initial_pos = obs.get('position', {})
                obs_vel = obs.get('velocity')
                obs_geometry = obs.get('geometry', {})
                obs_x_initial = obs_initial_pos.get('x', 0)
                obs_y_initial = obs_initial_pos.get('y', 0)
                if obs_vel:
                    obs_x_future = obs_x_initial + obs_vel.get('vx', 0) * time_at_point
                    obs_y_future = obs_y_initial + obs_vel.get('vy', 0) * time_at_point
                else:
                    obs_x_future = obs_x_initial
                    obs_y_future = obs_y_initial
                
                obs_r = 0.0
                if obs_geometry.get('type') == 'circle':
                    obs_r = obs_geometry.get('radius', 1.0)
                elif obs_geometry.get('type') == 'rectangle':
                    length = obs_geometry.get('length', 1.0)
                    width = obs_geometry.get('width', 1.0)
                    obs_r = math.sqrt((length/2)**2 + (width/2)**2)
                
                for offset in body_circle_offsets:
                    body_circle_x = robot_x + offset * math.cos(robot_theta)
                    body_circle_y = robot_y + offset * math.sin(robot_theta)
                    dist_sq = (body_circle_x - obs_x_future)**2 + (body_circle_y - obs_y_future)**2
                    dist_to_surface = math.sqrt(dist_sq) - obs_r - body_circle_radius
                    min_dist_to_any_obstacle_surface = min(min_dist_to_any_obstacle_surface, dist_to_surface)

        if min_dist_to_any_obstacle_surface >= current_safe_distance:
            return 0.0 
        
        if min_dist_to_any_obstacle_surface < 1e-3:
             cost = 1.0
        else:
             cost = (current_safe_distance - min_dist_to_any_obstacle_surface) / current_safe_distance
        
        cost = cost**0.7
        return max(0.0, min(cost, 1.0))

    def _get_type_based_safe_distance(self, obs_type, base_safe_distance):
        """
        根据障碍物类型返回相应的安全距离
        """
        type_multipliers = {
            'vessel': 1.5,      # 船舶需要更大安全距离
            'buoy': 0.8,        # 浮标可以稍微近一些
            'structure': 1.2,   # 固定结构中等安全距离
            'debris': 0.6,      # 漂浮物较小安全距离
            'unknown': 1.8      # 未知类型保守处理
        }
        
        multiplier = type_multipliers.get(obs_type, 1.0) #系数，根据障碍物类型调整安全距离
        return base_safe_distance * multiplier

    def normalize_angle(self, angle):
        """将角度归一化到 [-pi, pi] 范围内"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

# --- ConfigManager 和 ParamTuner 类保持不变 ---
# (假设它们在文件末尾，并且你想保留它们)
# ... (Insert ConfigManager and ParamTuner class code here if it was present and should be kept) ...

class SimulationVisualizer:
    """使用Tkinter进行DWA规划的可视化模拟"""
    def __init__(self, planner):
        self.planner = planner
        self.window = tk.Tk()
        self.window.title("DWA动态避障模拟器")

        # 画布尺寸和缩放比例
        self.canvas_width = 1000 # 500m * 2.0 scale
        self.canvas_height = 240 # 120m * 2.0 scale
        self.scale = 2.0 # 1米 = 2.0像素, 使得 500x120 米的世界映射到 1000x240 像素的画布
        
        # 从规划器的配置中获取船体尺寸
        self.robot_length = self.planner.base_config.get('robot_length', 10.0)
        self.robot_width = self.planner.base_config.get('robot_width', 3.0)

        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        # 初始化模拟状态 - 适应500x120米水平场景
        self.current_pos = [30.0, 60.0, 0.0] # 从左侧中间出发, 朝向正东
        self.current_vel = 0.0
        self.current_omega = 0.0
        self.goal = [470.0, 60.0] # 目标在右侧中间
        self.obstacles = [
            # --- 新的水平通道场景 ---
            {"id": 201, "type": "structure", "position": {"x": 100.0, "y": 30.0}, "geometry": {"type": "circle", "radius": 10.0}},
            {"id": 202, "type": "structure", "position": {"x": 250.0, "y": 90.0}, "geometry": {"type": "circle", "radius": 5.0}},
            {"id": 203, "type": "structure", "position": {"x": 150.0, "y": 55.0}, "geometry": {"type": "circle", "radius": 7.0}},
            {"id": 101, "type": "vessel", "position": {"x": 400.0, "y": 100.0}, "velocity": {"vx": -2.0, "vy": -0.5}, "geometry": {"type": "rectangle", "length": 30.0, "width": 10.0}},
            {"id": 102, "type": "vessel", "position": {"x": 300.0, "y": 20.0}, "velocity": {"vx": -2.0, "vy": 0.0}, "geometry": {"type": "rectangle", "length": 20.0, "width": 10.0}},
        ]
        
        # 时间步长
        self.dt = self.planner.base_config['dt'] # 与规划器保持一致
        self.simulation_running = True
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def determine_navigation_state(self):
        """
        一个简化的状态机，用于判断当前是否处于紧急避碰状态。
        """
        ship_x, ship_y, ship_theta = self.current_pos
        safe_dist = self.planner.base_config['base_safe_distance']
        
        # 危险检测的半径，比安全距离稍大
        detection_radius = safe_dist * 1.5 

        for obs in self.obstacles:
            obs_pos = obs.get('position', {})
            obs_x, obs_y = obs_pos.get('x', 0), obs_pos.get('y', 0)
            
            dist_sq = (ship_x - obs_x)**2 + (ship_y - obs_y)**2
            
            # 1. 只考虑足够近的障碍物
            if dist_sq < detection_radius**2:
                # 2. 判断障碍物是否在船头方向
                angle_to_obs = math.atan2(obs_y - ship_y, obs_x - ship_x)
                heading_diff = self.planner.normalize_angle(angle_to_obs - ship_theta)
                
                # 3. 如果在前方45度锥形区域内，则视为紧急情况
                if abs(heading_diff) < (np.pi / 4): 
                    # print(f"DEBUG: EMERGENCY STATE! Obstacle {obs.get('id')} ahead.")
                    return "EMERGENCY_AVOIDANCE_RIGHT"
                    
        return "NORMAL_NAVIGATION"

    def world_to_canvas(self, x, y):
        """将世界坐标转换为画布坐标"""
        # 世界坐标系原点在左下角，Y轴向上
        # 画布坐标系原点在左上角，Y轴向下
        return x * self.scale, self.canvas_height - y * self.scale

    def draw_robot(self):
        # 船体中心世界坐标和航向
        cx, cy, theta = self.current_pos[0], self.current_pos[1], self.current_pos[2]
        
        # 船体尺寸
        length = self.robot_length
        width = self.robot_width
        
        # 定义船体在自身坐标系下的四个角点（中心为原点）
        hl, hw = length / 2.0, width / 2.0
        corners_body_frame = [
            (hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)
        ]
        
        # 将角点旋转并平移到世界坐标系
        corners_world_frame = []
        for x, y in corners_body_frame:
            world_x = cx + x * np.cos(theta) - y * np.sin(theta)
            world_y = cy + x * np.sin(theta) + y * np.cos(theta)
            corners_world_frame.append((world_x, world_y))
            
        # 将世界坐标转换为画布坐标
        canvas_coords = []
        for wx, wy in corners_world_frame:
            px, py = self.world_to_canvas(wx, wy)
            canvas_coords.extend([px, py])
            
        # 绘制多边形代表船体
        self.canvas.create_polygon(canvas_coords, fill="blue", outline="black", tags="robot")

        # 绘制航向线 (从中心指向船头中点)
        canvas_cx, canvas_cy = self.world_to_canvas(cx, cy)
        front_x = cx + hl * np.cos(theta)
        front_y = cy + hl * np.sin(theta)
        canvas_front_x, canvas_front_y = self.world_to_canvas(front_x, front_y)
        self.canvas.create_line(canvas_cx, canvas_cy, canvas_front_x, canvas_front_y, fill="white", width=2, tags="robot")

    def draw_goal(self):
        x, y = self.world_to_canvas(self.goal[0], self.goal[1])
        r = 5
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="green", outline="black", tags="goal")
        self.canvas.create_text(x, y-10, text="Goal", fill="green", tags="goal")

    def draw_obstacles(self):
        # 绘制障碍物
        for obs in self.obstacles:
            # 获取通用属性
            obs_pos = obs.get('position', {'x': 0, 'y': 0})
            cx, cy = obs_pos['x'], obs_pos['y']
            obs_type = obs.get('type', 'unknown')
            obs_id = obs.get('id', '?')
            geometry = obs.get('geometry', {})
            
            # 根据类型设置颜色
            color_map = {
                'vessel': "red", 'buoy': "orange", 'structure': "brown", 
                'debris': "yellow", 'unknown': "gray"
            }
            color = color_map.get(obs_type, "gray")

            # 动态障碍物用更深的颜色
            vel_data = obs.get('velocity', {})
            if vel_data and (vel_data.get('vx', 0) != 0 or vel_data.get('vy', 0) != 0):
                color = "dark" + color if color != "gray" else "darkgray"

            # --- 根据几何形状绘制 ---
            if geometry.get('type') == 'rectangle':
                # 绘制旋转的矩形
                length = geometry.get('length', 10.0)
                width = geometry.get('width', 5.0)
                
                # 从速度计算朝向
                vx = vel_data.get('vx', 0)
                vy = vel_data.get('vy', 0)
                theta = math.atan2(vy, vx)

                # 定义角点并变换到画布坐标
                hl, hw = length / 2.0, width / 2.0
                corners_body = [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
                canvas_coords = []
                for x_b, y_b in corners_body:
                    world_x = cx + x_b * np.cos(theta) - y_b * np.sin(theta)
                    world_y = cy + x_b * np.sin(theta) + y_b * np.cos(theta)
                    px, py = self.world_to_canvas(world_x, world_y)
                    canvas_coords.extend([px, py])
                
                self.canvas.create_polygon(canvas_coords, fill=color, outline="black", tags="obstacle")
            
            elif geometry.get('type') == 'circle':
                # 绘制圆形
                r = geometry.get('radius', 1.0) * self.scale
                x_c, y_c = self.world_to_canvas(cx, cy)
                self.canvas.create_oval(x_c - r, y_c - r, x_c + r, y_c + r, fill=color, outline="black", tags="obstacle")
            
            # 显示ID和类型文本
            canvas_cx, canvas_cy = self.world_to_canvas(cx, cy)
            self.canvas.create_text(canvas_cx, canvas_cy, text=f"{obs_type}\nID:{obs_id}", fill="white", tags="obstacle")
    #绘制轨迹，轨迹是规划器计算出的最佳路径，用紫色虚线表示
    def draw_trajectory(self, trajectory):
        # 只有当轨迹点数大于等于2时才绘制
        if not trajectory or len(trajectory) < 2:
            return
        
        path_points = []
        # 遍历 trajectory 列表中的每一个 point
        for point in trajectory:
            px, py = self.world_to_canvas(point[0], point[1])
            path_points.extend([px, py])
        # 绘制轨迹，用紫色虚线表示
        self.canvas.create_line(path_points, fill="purple", width=2, dash=(2, 2), tags="trajectory")

    #   更新仿真世界的状态，并重绘整个场景
    def update_simulation(self):
        if not self.simulation_running:
            return

        # 1. 新增: 判断当前导航状态
        current_nav_state = self.determine_navigation_state()

        # 2. 调用DWA规划，传入当前状态
        best_v, best_omega, best_trajectory = self.planner.dwa_planning(
            self.current_pos, self.current_vel, self.current_omega, self.goal, self.obstacles, current_nav_state
        )

        # 3. 更新本船状态
        self.current_vel = best_v
        self.current_omega = best_omega
        # 使用简单的运动学模型（欧拉积分）更新位置和朝向
        self.current_pos[0] += self.current_vel * np.cos(self.current_pos[2]) * self.dt
        self.current_pos[1] += self.current_vel * np.sin(self.current_pos[2]) * self.dt
        self.current_pos[2] = self.planner.normalize_angle(self.current_pos[2] + self.current_omega * self.dt)

        # 4. 更新动态障碍物位置
        for obs in self.obstacles:
            if 'velocity' in obs:
                obs['position']['x'] += obs['velocity']['vx'] * self.dt
                obs['position']['y'] += obs['velocity']['vy'] * self.dt

        # 5. 重新绘制所有元素
        self.canvas.delete("all")
        self.draw_robot()
        self.draw_goal()
        self.draw_obstacles()
        self.draw_trajectory(best_trajectory)
        
        # 6. 检查是否到达目标
        dist_to_goal = np.sqrt((self.current_pos[0] - self.goal[0])**2 + (self.current_pos[1] - self.goal[1])**2)
        # 新的到达条件: 只要船体的任何部分在目标点2米范围内，即视为到达。
        # 我们通过检查船中心到目标的距离是否小于 (2米 + 船体长度的一半) 来近似判断。
        arrival_distance = 2.0 # 到达目标的距离 (米)
        if dist_to_goal < (arrival_distance + self.robot_length / 2.0):
            print("目标已到达！")
            self.simulation_running = False
            self.canvas.create_text(self.canvas_width/2, self.canvas_height/2, text="Goal Reached!", font=("Arial", 32), fill="green")

        # 7. 安排下一次更新
        self.window.after(int(self.dt * 1000), self.update_simulation)
        
    #  Tkinter 图形用户界面（GUI）应用程序的生命周期管理核心，负责启动、运行和安全地关闭仿真窗口。
    def on_close(self):
        """当用户关闭窗口时调用的方法"""
        # 1. 设置仿真运行标志为False
        self.simulation_running = False
        # 2. 销毁Tkinter窗口
        self.window.destroy()

    def run(self):
        """启动仿真和GUI的主方法"""
        # 1. 立即执行一次仿真更新
        self.update_simulation()
        # 2. 启动Tkinter的事件主循环
        self.window.mainloop()

# 如果作为主程序运行，进行简单测试
if __name__ == '__main__':
    print("DWA局部规划器 - 可视化模拟")
    
    # 创建一个为当前仿真场景特别优化的配置字典。
    # DWA算法的效果高度依赖于这些参数，针对不同环境或机器人进行调优是常见的做法。
    test_config = {
        # --- 动力学: 赋予算法足够的启动能力 ---
        'max_speed': 3.0,
        'min_speed': 0.0,
        'max_omega': 0.25,
        'min_omega': -0.25,
        'max_accel': 0.15,           # 核心: 一个相对真实但能让算法启动的加速度
        'max_domega': 0.05,
        
        # --- 决策: 目标导向，避免盘旋 ---
        'v_resolution': 0.01, 
        'omega_resolution': 0.01,   # 核心: 降低分辨率以匹配低加速度
        'dt': 0.1, 
        'predict_time': 8.0,        # 对于慢响应的船，需要更长的远见
        'base_heading_weight': 0.8, # 核心: 恢复部分航向权重以减少犹豫，提供更清晰的目标导向
        'base_dist_weight': 0.6,    # 核心: 鼓励靠近目标
        'base_velocity_weight': 0.5,  # 速度是次要的，正确方向的速度才是重要的
        'base_obstacle_weight': 2.5,  # 大幅提高障碍物权重，使其更谨慎
        'base_safe_distance': 25.0, # 增加安全距离，让船舶更早做出反应
        
        # 新的船体模型参数 - 根据船舶证书更新
        # 增加了1米的膨胀体积并向上取整
        'robot_length': 72.0,
        'robot_width': 16.0,
        'num_body_circles': 5, # 用5个圆来近似船体
    }
    planner = DWAPlanner(config=test_config)
    
    # 启动可视化模拟器
    visualizer = SimulationVisualizer(planner)
    visualizer.run()

# 保留 ConfigManager 和 ParamTuner 如果它们在原始文件中
# (确保这部分代码在最末尾，并且与上面的DWAPlanner类分离)

class ConfigManager:
    """
    配置管理器类。
    负责从YAML文件中加载和保存模块的配置参数。
    它提供了一个中心化的方式来处理配置，实现了配置与代码的分离。
    """
    
    def __init__(self, config_dir='config'):
        """
        初始化配置管理器。
        Args:
            config_dir (str): 存放配置文件的目录名。
        """
        self.config_dir = config_dir  # 存储配置文件的目录路径
        self.configs = {}  # 作为内存缓存，避免重复读取文件
        # 在初始化时，确保DWA的默认配置文件存在，如果不存在则创建一个。
        self._ensure_default_dwa_config()

    def _ensure_default_dwa_config(self):
        """
        一个内部辅助方法，用于检查并创建默认的DWA配置文件。
        如果不存在，它会定义一个包含标准DWA参数的字典，并调用 save_config 方法将其写入文件，同时打印提示信息。
        """
        module_name = 'dwa'
        config_path = os.path.join(self.config_dir, f"{module_name}_config.yaml")
        
        # 如果配置文件不存在
        if not os.path.exists(config_path):
            print(f"提示: 未找到DWA配置文件 {config_path}，将创建默认配置。")
            # 定义一套标准的、可用的DWA默认参数
            default_dwa_config = {
                'max_speed': 2.0, 'min_speed': 0.0, 'max_omega': 1.0, 'min_omega': -1.0,
                'max_accel': 0.5, 'max_domega': 1.0, 'v_resolution': 0.1, 'omega_resolution': 0.1,
                'dt': 0.1, 'predict_time': 3.0,
                'base_heading_weight': 0.8, 'base_dist_weight': 0.2,
                'base_velocity_weight': 0.1, 'base_obstacle_weight': 1.5,
                'base_safe_distance': 5.0,
            }
            # 调用保存方法，将默认配置写入文件
            self.save_config(module_name, default_dwa_config)


    def load_config(self, module_name):
        """
        从YAML文件加载指定模块的配置。
        Args:
            module_name (str): 模块名，如 'dwa'。
        Returns:
            dict: 加载到的配置字典，如果失败则返回空字典。
        """
        config_path = os.path.join(self.config_dir, f"{module_name}_config.yaml")
        
        # 鲁棒性处理：检查配置文件是否存在
        if not os.path.exists(config_path):
            print(f"警告: 配置文件 {config_path} 不存在。对于DWA，将尝试使用内部默认值。")
            # 对于DWA模块，其主类有内部默认值，所以返回空字典是安全的。
            if module_name == 'dwa':
                 return {} # 返回空字典，DWAPlanner.__init__会使用其内置默认
            return {} 
            
        # 使用try-except块来安全地处理文件读取和解析过程中的潜在错误
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 鲁棒性处理：检查文件是否为空或格式无效
                if config is None:
                    print(f"警告: 配置文件 {config_path} 为空或无效。")
                    return {}
                # 加载成功后，存入缓存
                self.configs[module_name] = config
                return config
        except Exception as e:
            print(f"加载配置文件 {config_path} 出错: {e}")
            return {}  # 发生任何错误都返回一个空字典，保证程序不会崩溃
    
    def get_config(self, module_name):
        """
        获取指定模块的配置。这是外部获取配置的主要接口。
        它实现了缓存逻辑，只有在缓存中不存在时才从文件加载。
        """
        # 如果配置不在缓存中
        if module_name not in self.configs:
            # 则从文件加载
            loaded_cfg = self.load_config(module_name)
            # 无论加载成功（得到配置字典）还是失败（得到空字典），都存入缓存
            # 这样可以避免对不存在或有误的配置文件进行重复的加载尝试。
            self.configs[module_name] = loaded_cfg 
            return loaded_cfg
        # 如果配置已在缓存中，直接返回
        return self.configs[module_name]
    
    def save_config(self, module_name, config):
        """
        将配置字典保存到YAML文件。
        Args:
            module_name (str): 模块名。
            config (dict): 要保存的配置字典。
        Returns:
            bool: 保存成功返回True, 失败返回False。
        """
        config_path = os.path.join(self.config_dir, f"{module_name}_config.yaml")
        
        try:
            # 确保配置目录存在，如果不存在则创建它
            os.makedirs(self.config_dir, exist_ok=True)
            # 以写入模式打开文件
            with open(config_path, 'w', encoding='utf-8') as f:
                # 使用yaml.dump写入。default_flow_style=False使格式更易读（块样式）
                # sort_keys=False 保持字典中键的原始顺序
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            # 保存成功后，更新内存中的缓存
            self.configs[module_name] = config
            print(f"配置已保存到 {config_path}")
            return True
        except Exception as e:
            print(f"保存配置文件 {config_path} 出错: {e}")
            return False

class ParamTuner:
    
    """参数调整工具"""
    #  建立依赖：接收并保存与外部（ConfigManager和主程序）通信所需的接口。
    # 创建窗口：初始化tkinter主窗口并设置标题。
    # 绑定事件：优雅地处理窗口关闭事件，确保程序可以安全地保存状态。
    # 准备数据结构：建立一个字典来管理所有的参数输入框。
    # 启动UI构建：调用另一个方法来动态生成具体的参数调整界面。
    def __init__(self, config_manager, on_close_callback=None): # 添加回调
        self.config_manager = config_manager
        self.on_close_callback = on_close_callback # 保存回调
        
        self.root = tk.Tk() # 使用tkinter库创建一个GUI应用程序的主窗口
        self.root.title("导航参数调整工具 (DWA)") # 特指DWA
        
        # 当窗口关闭时调用
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.param_entries = {} # 用于存储Entry控件
        self.module_name = 'dwa' # 这个调参器目前只针对DWA

        self.create_dwa_tab(self.module_name, 'DWA规划器参数')
        
        # 添加全局保存按钮
        # save_all_btn = tk.Button(self.root, text="保存DWA配置", command=self.save_current_dwa_config)
        # save_all_btn.pack(pady=10, side=tk.BOTTOM)

    # 当窗口关闭时调用
    def _on_closing(self):
        # print("调参工具关闭。")
        if self.on_close_callback:
            self.on_close_callback(self.get_current_config_from_gui())
        self.root.destroy()

    def get_current_config_from_gui(self):
        """从GUI获取当前参数值 并将这些值打包成一个配置字典"""
        current_config = {}
        for key, entry_var in self.param_entries.items():
            value_str = entry_var.get()
            try:
                if '.' in value_str or 'e' in value_str.lower():
                    value = float(value_str)
                else:
                    value = int(value_str)
            except ValueError:
                value = value_str #保持字符串
            current_config[key] = value
        return current_config

    def create_dwa_tab(self, module_name, tab_title):
        """创建DWA参数调整界面 (不再使用Tab，直接在主窗口)
        智能加载配置，优先使用用户自定义的文件配置。
        自动遍历参数，无需手动为每个参数编写UI代码。
        动态创建标签和输入框，并将它们整齐地排列。
        建立数据链接，通过 StringVar 和 param_entries 字典，为后续的数据读取和保存做好了准备。
        提供明确的操作，通过一个功能清晰的保存按钮完成闭环。
        """
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(expand=True, fill="both")
        
        # 获取模块配置 (确保获取的是最新的，或者DWAPlanner使用的基础配置)
        # 我们将基于DWAPlanner的内置默认值来填充GUI
        temp_planner = DWAPlanner() # 获取其base_config
        config = temp_planner.base_config 
        # 也尝试从文件加载，如果文件存在，则优先使用文件中的
        loaded_config_from_file = self.config_manager.get_config(module_name)
        if loaded_config_from_file: # 如果加载到了有效配置
            config.update(loaded_config_from_file)


        row = 0
        for key, value in sorted(config.items()): # 按键排序显示
            label = ttk.Label(frame, text=f"{key}:")
            label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            
            entry_var = tk.StringVar(value=str(value))
            entry = ttk.Entry(frame, textvariable=entry_var, width=25)
            entry.grid(row=row, column=1, padx=5, pady=2)
            
            self.param_entries[key] = entry_var # 存储StringVar
            row += 1
        
        # 保存按钮
        save_btn = ttk.Button(
            frame, 
            text=f"保存到 {module_name}_config.yaml", 
            command=self.save_current_dwa_config_to_file
        )
        save_btn.grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
    
    #   当用户点击界面上那个 "保存到 ... .yaml" 按钮时，调用该函数。
    def save_current_dwa_config_to_file(self):
        """保存当前GUI中的DWA配置到文件"""
        config_to_save = self.get_current_config_from_gui()
        if self.config_manager.save_config(self.module_name, config_to_save):
            print(f"DWA配置已成功保存到文件。")
        else:
            print(f"DWA配置保存失败。")

    def run(self):
        self.root.mainloop()
        
#   仅当这个脚本是作为主程序被直接运行时，才执行我下方的代码块
if __name__ == "__main__":
    # DWAPlanner 测试部分已在上面，这里是 ConfigManager 和 ParamTuner 的测试
    print("\n--- ConfigManager 和 ParamTuner 测试 ---")
    
    # 确保测试时 config 目录存在，或 ConfigManager 能正确处理
    if not os.path.exists("config"):
        os.makedirs("config")
        print("创建 'config' 目录用于测试。")

    config_manager = ConfigManager(config_dir='config') # 指定目录
    
    # 尝试加载 'dwa' 配置 (此时可能会创建默认的dwa_config.yaml)
    dwa_cfg = config_manager.get_config('dwa')
    print(f"从ConfigManager加载的DWA初始配置: {dwa_cfg}")

    # 创建DWAPlanner实例，它会使用ConfigManager加载的配置（如果存在）
    # 或者使用其内部定义的默认值（如果ConfigManager未返回有效配置）
    # planner_for_tuner = DWAPlanner(config_manager=config_manager)


    # 定义一个简单的回调，当调参工具关闭时打印获取到的参数
    def tuner_closed_callback(updated_config):
        print("\nParamTuner已关闭。从GUI获取的最终DWA参数为:")
        for key, value in updated_config.items():
            print(f"  {key}: {value} (类型: {type(value).__name__})")
        # 在这里，你可以选择用这些 updated_config 来重新配置你的 DWAPlanner 实例
        # 例如: planner_for_tuner.base_config.update(updated_config)
        # print("DWAPlanner 的 base_config 已用GUI参数更新 (如果需要)。")


    print("启动参数调整工具 ParamTuner (仅针对DWA)...")
    tuner = ParamTuner(config_manager, on_close_callback=tuner_closed_callback)
    tuner.run() # 这会阻塞，直到GUI窗口关闭

    print("ParamTuner 测试结束。") 