import tkinter as tk
from tkinter import ttk
import numpy as np
import math

# 从其他模块导入必要的类
from dwa_planner import DWAPlanner
from config_manager import ConfigManager

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