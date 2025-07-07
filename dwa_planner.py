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
        if current_state == self.base_config.get('DWA_CMD_AVOID', "AVOIDANCE_MANEUVER"):
            active_obstacle_weight *= 2.0  # 提高避障权重
            active_heading_weight *= 0.5   # 降低对原始航向的执着
            active_velocity_weight *= 0.8  # 允许适当减速
            active_safe_distance *= 1.2    # 增加安全距离
            print("INFO: DWA State: AVOIDANCE_MANEUVER")

        elif current_state == self.base_config.get('DWA_CMD_RETURN', "RETURN_TO_PATH"):
            active_heading_weight *= 1.5   # 强烈鼓励朝向目标点
            active_obstacle_weight *= 1.2  # 在返回时也要注意障碍物
            active_velocity_weight *= 1.1  # 鼓励尽快恢复速度
            print("INFO: DWA State: RETURN_TO_PATH")

        elif current_state == self.base_config.get('DWA_CMD_STOP', "EMERGENCY_STOP"):
            print("INFO: DWA State: EMERGENCY_STOP, returning (0,0)")
            return 0.0, 0.0, [current_pos[:2]] # 立即停止

        elif current_state == self.base_config.get('DWA_CMD_FAILURE', "CAUTIOUS_NAVIGATION"):
            # 在故障或谨慎模式下，降低最大速度，并使用保守权重
            max_speed *= 0.5 # 将最大速度减半
            active_obstacle_weight *= 1.5
            active_velocity_weight = 0.0 # 不再鼓励高速
            print("INFO: DWA State: CAUTIOUS_NAVIGATION")

        elif current_state == self.base_config.get('DWA_CMD_NORMAL', "NORMAL_NAVIGATION"):
            # print(f"INFO: DWA State: NORMAL_NAVIGATION")
            pass # 使用基础参数
        else:
            print(f"WARN: DWA received unknown state '{current_state}'. Using NORMAL_NAVIGATION parameters.")
            pass # 未知状态，使用默认参数

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

        # 遍历动态窗口中的所有速度对 (v, omega)
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

            trajectory.append([current_x, current_y])
        
        return trajectory

    def calc_obstacle_cost(self, trajectory, obstacles, current_safe_distance, dt):
        """
        计算轨迹与障碍物的代价（支持v2.0接口的动态障碍物）。
        代价范围: 0 (完全安全) 到 1 (非常接近安全边界但未碰撞)。
        如果碰撞，则返回 float('inf')。
        """
        if not obstacles:
            return 0.0 # 没有障碍物，代价为0

        min_dist_to_any_obstacle_surface = float('inf')

        # 遍历自己轨迹的每个点
        for i, point in enumerate(trajectory):
            # 计算该轨迹点对应的时间
            time_at_point = (i + 1) * dt

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
                    
                    # 处理几何形状
                    if obs_geometry.get('type') == 'circle':
                        obs_r = obs_geometry.get('radius', 1.0)
                        # 使用圆形碰撞检测
                        dist_sq = (point[0] - obs_x_future)**2 + (point[1] - obs_y_future)**2
                        
                        # 根据障碍物类型调整安全距离
                        type_safe_distance = self._get_type_based_safe_distance(obs_type, current_safe_distance)
                        
                        if dist_sq <= obs_r**2:
                            return float('inf') # 发生碰撞
                        
                        dist_to_surface = math.sqrt(dist_sq) - obs_r
                        min_dist_to_any_obstacle_surface = min(min_dist_to_any_obstacle_surface, dist_to_surface)
                        
                    elif obs_geometry.get('type') == 'rectangle':
                        # 矩形障碍物处理（简化为外接圆）
                        length = obs_geometry.get('length', 10.0)
                        width = obs_geometry.get('width', 5.0)
                        # 使用外接圆半径作为近似
                        obs_r = math.sqrt((length/2)**2 + (width/2)**2)
                        
                        dist_sq = (point[0] - obs_x_future)**2 + (point[1] - obs_y_future)**2
                        
                        # 根据障碍物类型调整安全距离
                        type_safe_distance = self._get_type_based_safe_distance(obs_type, current_safe_distance)
                        
                        if dist_sq <= obs_r**2:
                            return float('inf') # 发生碰撞
                        
                        dist_to_surface = math.sqrt(dist_sq) - obs_r
                        min_dist_to_any_obstacle_surface = min(min_dist_to_any_obstacle_surface, dist_to_surface)
                    else:
                        # 兼容旧版本或默认处理
                        obs_r = obs.get('radius', 1.0)
                        dist_sq = (point[0] - obs_x_future)**2 + (point[1] - obs_y_future)**2
                        
                        if dist_sq <= obs_r**2:
                            return float('inf')
                        
                        dist_to_surface = math.sqrt(dist_sq) - obs_r
                        min_dist_to_any_obstacle_surface = min(min_dist_to_any_obstacle_surface, dist_to_surface)
                
                elif isinstance(obs, (list, tuple)) and len(obs) == 3:
                    # --- 旧格式：[x, y, r]，仅支持静态障碍物 ---
                    obs_x_future, obs_y_future, obs_r = obs
                    
                    dist_sq = (point[0] - obs_x_future)**2 + (point[1] - obs_y_future)**2
                    
                    if dist_sq <= obs_r**2:
                        return float('inf')
                    
                    dist_to_surface = math.sqrt(dist_sq) - obs_r
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
        
        multiplier = type_multipliers.get(obs_type, 1.0)
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
        self.scale = 20 # 1米 = 20像素
        self.robot_radius_px = 0.5 * self.scale # 机器人显示半径

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
        x, y = self.world_to_canvas(self.current_pos[0], self.current_pos[1])
        theta = self.current_pos[2]
        
        # 船体
        self.canvas.create_oval(x - self.robot_radius_px, y - self.robot_radius_px,
                                x + self.robot_radius_px, y + self.robot_radius_px,
                                fill="blue", outline="black", tags="robot")
        # 航向线
        end_x = x + self.robot_radius_px * np.cos(theta)
        end_y = y - self.robot_radius_px * np.sin(theta) # 画布Y轴向下，用减号
        self.canvas.create_line(x, y, end_x, end_y, fill="white", width=2, tags="robot")

    def draw_goal(self):
        x, y = self.world_to_canvas(self.goal[0], self.goal[1])
        r = 5
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="green", outline="black", tags="goal")
        self.canvas.create_text(x, y-10, text="Goal", fill="green", tags="goal")

    def draw_obstacles(self):
        for obs in self.obstacles:
            # 兼容v2.0和旧格式
            if isinstance(obs, dict):
                # v2.0格式
                if 'geometry' in obs:
                    geometry = obs['geometry']
                    if geometry.get('type') == 'circle':
                        r = geometry.get('radius', 1.0) * self.scale
                    elif geometry.get('type') == 'rectangle':
                        # 矩形用外接圆近似显示
                        length = geometry.get('length', 10.0)
                        width = geometry.get('width', 5.0)
                        r = math.sqrt((length/2)**2 + (width/2)**2) * self.scale
                    else:
                        r = 5.0 * self.scale  # 默认半径
                elif 'radius' in obs:
                    # 旧v1.0格式
                    r = obs['radius'] * self.scale
                else:
                    r = 5.0 * self.scale  # 默认半径
                
                obs_pos = obs.get('position', {'x': 0, 'y': 0})
                x, y = self.world_to_canvas(obs_pos['x'], obs_pos['y'])
                
                # 根据类型设置颜色
                obs_type = obs.get('type', 'unknown')
                if obs_type == 'vessel':
                    color = "red"
                elif obs_type == 'buoy':
                    color = "orange"
                elif obs_type == 'structure':
                    color = "brown"
                elif obs_type == 'debris':
                    color = "yellow"
                else:
                    color = "gray"
                
                # 动态障碍物用更深的颜色
                if 'velocity' in obs and (obs['velocity']['vx'] != 0 or obs['velocity']['vy'] != 0):
                    color = "dark" + color if color != "gray" else "darkgray"
                
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

    def draw_trajectory(self, trajectory):
        # 只有当轨迹点数大于等于2时才绘制
        if not trajectory or len(trajectory) < 2:
            return
        
        path_points = []
        for point in trajectory:
            px, py = self.world_to_canvas(point[0], point[1])
            path_points.extend([px, py])
            
        self.canvas.create_line(path_points, fill="purple", width=2, dash=(2, 2), tags="trajectory")

    def update_simulation(self):
        if not self.simulation_running:
            return

        # 1. 调用DWA规划
        best_v, best_omega, best_trajectory = self.planner.dwa_planning(
            self.current_pos, self.current_vel, self.current_omega, self.goal, self.obstacles, "NORMAL_NAVIGATION"
        )

        # 2. 更新本船状态
        self.current_vel = best_v
        self.current_omega = best_omega
        # 使用简单的运动学模型更新位置和朝向
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

    def on_close(self):
        self.simulation_running = False
        self.window.destroy()

    def run(self):
        self.update_simulation()
        self.window.mainloop()

# 如果作为主程序运行，进行简单测试
if __name__ == '__main__':
    print("DWA局部规划器 - 可视化模拟")
    
    # 创建一个简单的配置字典用于测试
    test_config = {
        'max_speed': 1.8, 'min_speed': 0.0, 'max_omega': 1.2, 'min_omega': -1.2,
        'max_accel': 0.8, 'max_domega': 1.5, 'v_resolution': 0.05, 'omega_resolution': 0.05,
        'dt': 0.1, 'predict_time': 1.8, # 从3.0缩短，让规划器更关注眼前，减少"冻结"
        'base_heading_weight': 0.6, 'base_dist_weight': 0.25, 
        'base_velocity_weight': 0.6, # 从0.15提升，鼓励机器人移动而不是停止
        'base_obstacle_weight': 1.7, # 从2.0略微降低，减少过度规避
        'base_safe_distance': 1.4 # 从2.0减小，允许通过更狭窄的通道
    }
    planner = DWAPlanner(config=test_config)
    
    # 启动可视化模拟器
    visualizer = SimulationVisualizer(planner)
    visualizer.run()

# 保留 ConfigManager 和 ParamTuner 如果它们在原始文件中
# (确保这部分代码在最末尾，并且与上面的DWAPlanner类分离)

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_dir='config'):
        self.config_dir = config_dir
        self.configs = {}
        # 自动为dwa创建一个默认的示例配置文件，如果config目录和文件不存在
        self._ensure_default_dwa_config()

    def _ensure_default_dwa_config(self):
        module_name = 'dwa'
        config_path = os.path.join(self.config_dir, f"{module_name}_config.yaml")
        if not os.path.exists(config_path):
            print(f"提示: 未找到DWA配置文件 {config_path}，将创建默认配置。")
            default_dwa_config = {
                'max_speed': 2.0, 'min_speed': 0.0, 'max_omega': 1.0, 'min_omega': -1.0,
                'max_accel': 0.5, 'max_domega': 1.0, 'v_resolution': 0.1, 'omega_resolution': 0.1,
                'dt': 0.1, 'predict_time': 3.0,
                'base_heading_weight': 0.8, 'base_dist_weight': 0.2,
                'base_velocity_weight': 0.1, 'base_obstacle_weight': 1.5,
                'base_safe_distance': 5.0,
            }
            self.save_config(module_name, default_dwa_config)


    def load_config(self, module_name):
        config_path = os.path.join(self.config_dir, f"{module_name}_config.yaml")
        
        if not os.path.exists(config_path):
            print(f"警告: 配置文件 {config_path} 不存在。对于DWA，将尝试使用内部默认值。")
            if module_name == 'dwa': # 特定处理DWA，因为它有内部默认
                 return {} # 返回空字典，DWAPlanner.__init__会使用其内置默认
            return {} 
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config is None: # YAML文件为空或解析结果为None
                    print(f"警告: 配置文件 {config_path} 为空或无效。")
                    return {}
                self.configs[module_name] = config
                return config
        except Exception as e:
            print(f"加载配置文件 {config_path} 出错: {e}")
            return {}
    
    def get_config(self, module_name):
        if module_name not in self.configs:
            loaded_cfg = self.load_config(module_name)
            # 如果load_config因为文件不存在或错误而返回空字典，
            # 确保configs中也存的是这个空字典，避免重复加载。
            self.configs[module_name] = loaded_cfg 
            return loaded_cfg
        return self.configs[module_name]
    
    def save_config(self, module_name, config):
        config_path = os.path.join(self.config_dir, f"{module_name}_config.yaml")
        
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            self.configs[module_name] = config
            print(f"配置已保存到 {config_path}")
            return True
        except Exception as e:
            print(f"保存配置文件 {config_path} 出错: {e}")
            return False

class ParamTuner:
    """参数调整工具"""
    
    def __init__(self, config_manager, on_close_callback=None): # 添加回调
        self.config_manager = config_manager
        self.on_close_callback = on_close_callback # 保存回调
        
        self.root = tk.Tk()
        self.root.title("导航参数调整工具 (DWA)") # 特指DWA
        
        # 当窗口关闭时调用
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.param_entries = {} # 用于存储Entry控件
        self.module_name = 'dwa' # 这个调参器目前只针对DWA

        self.create_dwa_tab(self.module_name, 'DWA规划器参数')
        
        # 添加全局保存按钮
        # save_all_btn = tk.Button(self.root, text="保存DWA配置", command=self.save_current_dwa_config)
        # save_all_btn.pack(pady=10, side=tk.BOTTOM)
        
    def _on_closing(self):
        # print("调参工具关闭。")
        if self.on_close_callback:
            self.on_close_callback(self.get_current_config_from_gui())
        self.root.destroy()

    def get_current_config_from_gui(self):
        """从GUI获取当前参数值"""
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
        """创建DWA参数调整界面 (不再使用Tab，直接在主窗口)"""
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
    
    def save_current_dwa_config_to_file(self):
        """保存当前GUI中的DWA配置到文件"""
        config_to_save = self.get_current_config_from_gui()
        if self.config_manager.save_config(self.module_name, config_to_save):
            print(f"DWA配置已成功保存到文件。")
        else:
            print(f"DWA配置保存失败。")

    def run(self):
        self.root.mainloop()

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