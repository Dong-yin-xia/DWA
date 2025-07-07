#!/usr/bin/env python3
import numpy as np
import math
# import rclpy # Removed as it's not used directly in the planner logic here
# from rclpy.node import Node # Removed
import yaml
import os
import tkinter as tk
from tkinter import ttk

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
                'max_speed': 2.0,         # 最大线速度 (m/s)
                'min_speed': 0.0,         # 最小线速度 (m/s) (可以是负数，如果允许倒退)
                'max_omega': 1.0,         # 最大角速度 (rad/s)
                'min_omega': -1.0,        # 最小角速度 (rad/s)
                'max_accel': 0.5,         # 最大线加速度 (m/s^2)
                'max_domega': 1.0,        # 最大角加速度 (rad/s^2)
                'v_resolution': 0.1,      # 线速度采样分辨率 (m/s)
                'omega_resolution': 0.1,  # 角速度采样分辨率 (rad/s)
                'dt': 0.1,                # 控制命令的时间步长 (s)
                'predict_time': 3.0,      # 轨迹预测时间 (s)
                'robot_length': 70.0,     # 机器人长度 (m)
                'robot_width': 3.0,       # 机器人宽度 (m)
                'base_heading_weight': 0.8,    # 基础航向权重
                'base_dist_weight': 0.2,       # 基础目标距离权重
                'base_velocity_weight': 0.1,   # 基础速度权重
                'base_obstacle_weight': 1.5,   # 基础障碍物权重 (之前是1.0，稍微提高)
                'base_safe_distance': 5.0,     # 基础安全距离 (m)
                # 'min_obstacle_dist': 1.0  # This seemed redundant with safe_distance, removing for now
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
            active_velocity_weight = 0.0 # 强烈倾向于低速或停止
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
                    current_pos[0], current_pos[1], current_theta, 
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

    def _dist_point_to_segment(self, p, v, w):
        """计算点p到线段vw的最短距离"""
        px, py = p
        vx, vy = v
        wx, wy = w
        
        l2 = (vx - wx)**2 + (vy - wy)**2
        if l2 == 0.0:
            return math.sqrt((px - vx)**2 + (py - vy)**2)
        
        t = max(0, min(1, ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2))
        
        proj_x = vx + t * (wx - vx)
        proj_y = vy + t * (wy - vy)
        
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def calc_obstacle_cost(self, trajectory, obstacles, current_safe_distance, dt):
        """
        计算轨迹与障碍物的代价（支持v2.0接口的动态障碍物）。
        代价范围: 0 (完全安全) 到 1 (非常接近安全边界但未碰撞)。
        如果碰撞，则返回 float('inf')。
        """
        robot_length = self.base_config.get('robot_length', 1.0)
        robot_width = self.base_config.get('robot_width', 0.5)
        robot_collision_radius = robot_width / 2.0
        half_length = robot_length / 2.0

        if not obstacles:
            return 0.0 # 没有障碍物，代价为0

        min_dist_to_any_obstacle_surface = float('inf')

        # 遍历自己轨迹的每个点
        for i, pose in enumerate(trajectory):
            px, py, p_theta = pose
            time_at_point = (i + 1) * dt

            # 计算机器人中心线段的端点
            cos_theta = math.cos(p_theta)
            sin_theta = math.sin(p_theta)
            p1 = (px - half_length * cos_theta, py - half_length * sin_theta)
            p2 = (px + half_length * cos_theta, py + half_length * sin_theta)


            # 检查与每个障碍物的距离
            for obs in obstacles:
                # 兼容旧的列表格式 [x, y, r]，并支持新的v2.0字典格式
                if isinstance(obs, dict):
                    # --- v2.0格式：字典，支持动态和静态障碍物 ---
                    obs_initial_pos = obs.get('position', {})
                    obs_vel = obs.get('velocity') # 如果没有velocity键，则为None
                    obs_type = obs.get('type', 'unknown')
                    obs_geometry = obs.get('geometry', {})
                    
                    obs_x_initial = obs_initial_pos.get('x', 0)
                    obs_y_initial = obs_initial_pos.get('y', 0)

                    # 如果是动态障碍物，预测其未来位置
                    if obs_vel:
                        obs_vx = obs_vel.get('vx', 0)
                        obs_vy = obs_vel.get('vy', 0)
                        obs_x_future = obs_x_initial + obs_vx * time_at_point
                        obs_y_future = obs_y_initial + obs_vy * time_at_point
                    else: # 静态障碍物
                        obs_x_future = obs_x_initial
                        obs_y_future = obs_y_initial
                    
                    obs_center_future = (obs_x_future, obs_y_future)

                    # 处理几何形状
                    if obs_geometry.get('type') == 'circle':
                        obs_r = obs_geometry.get('radius', 1.0)  # 忘记提供，默认障碍物半径
                        
                        dist_to_centerline = self._dist_point_to_segment(obs_center_future, p1, p2)
                        
                        # 根据障碍物类型调整安全距离
                        type_safe_distance = self._get_type_based_safe_distance(obs_type, current_safe_distance)
                        
                        if dist_to_centerline <= obs_r + robot_collision_radius:
                            return float('inf') # 发生碰撞
                        
                        dist_to_surface = dist_to_centerline - obs_r - robot_collision_radius
                        min_dist_to_any_obstacle_surface = min(min_dist_to_any_obstacle_surface, dist_to_surface)
                        
                    elif obs_geometry.get('type') == 'rectangle':
                        # 矩形障碍物处理（简化为外接圆）
                        length = obs_geometry.get('length', 10.0)
                        width = obs_geometry.get('width', 5.0)
                        # 使用外接圆半径作为近似
                        obs_r = math.sqrt((length/2)**2 + (width/2)**2)
                        
                        dist_to_centerline = self._dist_point_to_segment(obs_center_future, p1, p2)
                        
                        # 根据障碍物类型调整安全距离
                        type_safe_distance = self._get_type_based_safe_distance(obs_type, current_safe_distance)
                        
                        if dist_to_centerline <= obs_r + robot_collision_radius:
                            return float('inf') # 发生碰撞
                        
                        dist_to_surface = dist_to_centerline - obs_r - robot_collision_radius
                        min_dist_to_any_obstacle_surface = min(min_dist_to_any_obstacle_surface, dist_to_surface)
                    else:
                        # 兼容旧版本或默认处理
                        obs_r = obs.get('radius', 1.0)
                        dist_to_centerline = self._dist_point_to_segment(obs_center_future, p1, p2)
                        
                        if dist_to_centerline <= obs_r + robot_collision_radius:
                            return float('inf')
                        
                        dist_to_surface = dist_to_centerline - obs_r - robot_collision_radius
                        min_dist_to_any_obstacle_surface = min(min_dist_to_any_obstacle_surface, dist_to_surface)
                
                elif isinstance(obs, (list, tuple)) and len(obs) == 3:
                    # --- 旧格式：[x, y, r]，仅支持静态障碍物 ---
                    obs_x_future, obs_y_future, obs_r = obs
                    obs_center_future = (obs_x_future, obs_y_future)

                    dist_to_centerline = self._dist_point_to_segment(obs_center_future, p1, p2)

                    if dist_to_centerline <= obs_r + robot_collision_radius:
                        return float('inf')
                    
                    dist_to_surface = dist_to_centerline - obs_r - robot_collision_radius
                    min_dist_to_any_obstacle_surface = min(min_dist_to_any_obstacle_surface, dist_to_surface)
                else:
                    # 不支持的障碍物格式，跳过
                    continue

        # 轨迹全程无碰撞，根据最小距离计算代价
        # 如果轨迹全程在安全距离之外
        if min_dist_to_any_obstacle_surface >= current_safe_distance:
            return 0.0 
        
        # 如果最小距离在 (0, current_safe_distance) 之间
        # 代价从0（在安全距离处）线性增加到接近1（在障碍物表面附近）
        if min_dist_to_any_obstacle_surface < 1e-3: # 非常接近或在表面
             cost = 1.0
        else:
             cost = (current_safe_distance - min_dist_to_any_obstacle_surface) / current_safe_distance
        
        # 可以应用非线性变换，使得越接近障碍物，代价增长越快
        cost = cost**0.7 # 指数小于1，会在接近1时增长更快，接近0时增长更慢
        return max(0.0, min(cost, 1.0)) # 确保代价在 [0,1]
    
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
        self.canvas_width = 800
        self.canvas_height = 600
        self.scale = 5 # 1米 = 5像素, 因船体变大而缩小比例
        self.robot_length = self.planner.base_config.get('robot_length', 70.0)
        self.robot_width = self.planner.base_config.get('robot_width', 3.0)

        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        # 初始化模拟状态 - 更有挑战性的场景
        self.current_pos = [3.0, 3.0, np.pi / 4] # 从左下角出发
        self.current_vel = 0.0
        self.current_omega = 0.0
        self.goal = [35.0, 25.0] # 目标在右上角
        self.obstacles = [
            # --- "地狱级"难度场景 ---
            # 动态障碍物 (5个)
            {"id": 101, "position": {"x": 15.0, "y": 28.0}, "velocity": {"vx": 0.0, "vy": -1.0}, "radius": 1.5}, # 从上往下
            {"id": 102, "position": {"x": 38.0, "y": 18.0}, "velocity": {"vx": -1.2, "vy": 0.2}, "radius": 1.2}, # 从右往左上
            {"id": 103, "position": {"x": 8.0, "y": 12.0}, "velocity": {"vx": 1.0, "vy": 0.5}, "radius": 1.0},   # 从左往右上
            {"id": 104, "position": {"x": 30.0, "y": 2.0}, "velocity": {"vx": -0.5, "vy": 1.0}, "radius": 1.3},   # 从右下往左上
            {"id": 105, "position": {"x": 35.0, "y": 25.0}, "velocity": {"vx": -1.5, "vy": -0.2}, "radius": 0.8}, # 从目标点附近过来
            # 静态障碍物 (3个)
            {"id": 201, "position": {"x": 12.0, "y": 20.0}, "radius": 2.0},
            {"id": 202, "position": {"x": 25.0, "y": 10.0}, "radius": 2.5},
            {"id": 203, "position": {"x": 20.0, "y": 15.0}, "radius": 2.2} # 增加一个在中间的
        ]
        
        # 时间步长
        self.dt = self.planner.base_config['dt'] # 与规划器保持一致
        self.simulation_running = True
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def world_to_canvas(self, x, y):
        """将世界坐标转换为画布坐标"""
        # 世界坐标系原点在左下角，Y轴向上
        # 画布坐标系原点在左上角，Y轴向下
        return x * self.scale, self.canvas_height - y * self.scale

    def draw_robot(self):
        cx, cy = self.world_to_canvas(self.current_pos[0], self.current_pos[1])
        theta = self.current_pos[2]
        
        length_px = self.robot_length * self.scale
        width_px = self.robot_width * self.scale
        
        hl = length_px / 2.0
        hw = width_px / 2.0
        
        # 定义矩形的四个角点（在机器人局部坐标系中）
        local_corners = [
            (hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)
        ]
        
        world_corners = []
        for x_local, y_local in local_corners:
            # 旋转
            x_rot = x_local * math.cos(theta) - y_local * math.sin(theta)
            y_rot = x_local * math.sin(theta) + y_local * math.cos(theta)
            
            # 平移并转换到画布坐标 (注意y轴反转)
            world_corners.append(cx + x_rot)
            world_corners.append(cy - y_rot)
            
        self.canvas.create_polygon(world_corners, fill="blue", outline="black", tags="robot")

    def draw_goal(self):
        x, y = self.world_to_canvas(self.goal[0], self.goal[1])
        r = 5
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="green", outline="black", tags="goal")
        self.canvas.create_text(x, y-10, text="Goal", fill="green", tags="goal")

    def draw_obstacles(self):
        # 绘制障碍物
        for obs in self.obstacles:
            # 兼容v2.0和旧格式
            if isinstance(obs, dict):
                # v2.0格式
                if 'geometry' in obs:
                    geometry = obs['geometry'] # 获取 'geometry' 键对应的值
                    if geometry.get('type') == 'circle':    # 圆形障碍物
                        r = geometry.get('radius', 1.0) * self.scale # 获取障碍物半径
                    elif geometry.get('type') == 'rectangle': # 矩形障碍物
                        # 矩形用外接圆近似显示
                        length = geometry.get('length', 10.0)
                        width = geometry.get('width', 5.0)
                        r = math.sqrt((length/2)**2 + (width/2)**2) * self.scale # 计算外接圆半径
                    else: # 障碍物有 geometry 字段，但其 type 既不是 'circle' 也不是 'rectangle'，就会执行这里
                        r = 5.0 * self.scale  # 默认半径
                elif 'radius' in obs:
                    #一个障碍物是字典格式，但没有 'geometry' 字段，程序就会检查这个条件。它会看这个字典的顶层是否直接包含一个 'radius' 键
                    # 旧v1.0格式
                    r = obs['radius'] * self.scale #直接使用顶层的半径值，并将其转换为像素半径。
                else:
                    r = 5.0 * self.scale  # 默认半径,没有 'geometry' 字段，也没有顶层的 'radius' 字段
                
                #从障碍物字典 obs 中获取其位置信息
                obs_pos = obs.get('position', {'x': 0, 'y': 0})
                # 获取障碍物的世界坐标，并将其转换为画布坐标
                x, y = self.world_to_canvas(obs_pos['x'], obs_pos['y'])
                
                # 根据类型设置颜色
                obs_type = obs.get('type', 'unknown')
                if obs_type == 'vessel': # 船舶
                    color = "red"
                elif obs_type == 'buoy': # 浮标
                    color = "orange"
                elif obs_type == 'structure': # 固定结构
                    color = "brown"
                elif obs_type == 'debris': # 漂浮物
                    color = "yellow"
                else: # 未知类型
                    color = "gray"
                
                # 动态障碍物用更深的颜色
                if 'velocity' in obs and (obs['velocity']['vx'] != 0 or obs['velocity']['vy'] != 0):
                    color = "dark" + color if color != "gray" else "darkgray"
                # 绘制障碍物的形状
                self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="black", tags="obstacle")
                
                # 显示ID和类型
                obs_id = obs.get('id', '?')
                self.canvas.create_text(x, y, text=f"{obs_type}\nID:{obs_id}", fill="white", tags="obstacle")
            
            elif isinstance(obs, (list, tuple)) and len(obs) == 3:
                # 旧格式 [x, y, radius]
                obs_x, obs_y, obs_r = obs
                r = obs_r * self.scale
                x, y = self.world_to_canvas(obs_x, obs_y)
                
                self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="gray", outline="black", tags="obstacle")
                self.canvas.create_text(x, y, text="Static", fill="white", tags="obstacle")
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

        # 1. 调用DWA规划，计算最佳速度和轨迹
        best_v, best_omega, best_trajectory = self.planner.dwa_planning(
            self.current_pos, self.current_vel, self.current_omega, self.goal, self.obstacles, "NORMAL_NAVIGATION"
        )

        # 2. 更新本船状态
        self.current_vel = best_v
        self.current_omega = best_omega
        # 使用简单的运动学模型（欧拉积分）更新位置和朝向
        self.current_pos[0] += self.current_vel * np.cos(self.current_pos[2]) * self.dt
        self.current_pos[1] += self.current_vel * np.sin(self.current_pos[2]) * self.dt
        self.current_pos[2] = self.planner.normalize_angle(self.current_pos[2] + self.current_omega * self.dt)

        # 3. 更新动态障碍物位置
        for obs in self.obstacles:
            if 'velocity' in obs:
                obs['position']['x'] += obs['velocity']['vx'] * self.dt
                obs['position']['y'] += obs['velocity']['vy'] * self.dt

        # 4. 重新绘制所有元素
        self.canvas.delete("all")
        self.draw_robot()
        self.draw_goal()
        self.draw_obstacles()
        self.draw_trajectory(best_trajectory)
        
        # 5. 检查是否到达目标
        dist_to_goal = np.sqrt((self.current_pos[0] - self.goal[0])**2 + (self.current_pos[1] - self.goal[1])**2)
        if dist_to_goal < 1.0: # 到达目标的阈值
            print("目标已到达！")
            self.simulation_running = False
            self.canvas.create_text(self.canvas_width/2, self.canvas_height/2, text="Goal Reached!", font=("Arial", 32), fill="green")

        # 6. 安排下一次更新
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
        'max_speed': 1.8, 'min_speed': 0.0, 'max_omega': 1.2, 'min_omega': -1.2,
        'max_accel': 0.8, 'max_domega': 1.5, 'v_resolution': 0.05, 'omega_resolution': 0.05,
        'dt': 0.1, 'predict_time': 3.0, # 矩形船体需要更长的预判时间
        'robot_length': 70.0, 'robot_width': 3.0, # 机器人几何尺寸
        'base_heading_weight': 0.6, 'base_dist_weight': 0.25, 
        'base_velocity_weight': 0.6, # 从0.15提升，鼓励机器人移动而不是停止
        'base_obstacle_weight': 2.5, # 对于大型船只，避障权重需要更高
        'base_safe_distance': 15.0 # 船体大，安全距离也要显著增大
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
                'robot_length': 70.0, 'robot_width': 3.0,
                'base_heading_weight': 0.8, 'base_dist_weight': 0.2,
                'base_velocity_weight': 0.1, 'base_obstacle_weight': 1.5,
                'base_safe_distance': 15.0,
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