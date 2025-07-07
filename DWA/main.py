import os
import sys

from dwa_planner import DWAPlanner
from simulation_gui import SimulationVisualizer, ParamTuner
from config_manager import ConfigManager

def run_simulation():
    """运行DWA仿真"""
    print("DWA局部规划器 - 可视化模拟")
    
    # 创建一个为当前仿真场景特别优化的配置字典。
    # DWA算法的效果高度依赖于这些参数，针对不同环境或机器人进行调优是常见的做法。
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

def run_tuner():
    """运行参数调优工具"""
    print("\n--- ConfigManager 和 ParamTuner 测试 ---")
    
    # 确保测试时 config 目录存在，或 ConfigManager 能正确处理
    if not os.path.exists("config"):
        os.makedirs("config")
        print("创建 'config' 目录用于测试。")

    config_manager = ConfigManager(config_dir='config') # 指定目录
    
    # 尝试加载 'dwa' 配置 (此时可能会创建默认的dwa_config.yaml)
    dwa_cfg = config_manager.get_config('dwa')
    print(f"从ConfigManager加载的DWA初始配置: {dwa_cfg}")

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

if __name__ == '__main__':
    """
    主程序入口。
    默认运行DWA仿真。
    可以通过命令行参数 'tuner' 来启动参数调整工具。
    
    用法:
    - 运行仿真: python main.py
    - 运行调参器: python main.py tuner
    """
    if len(sys.argv) > 1 and sys.argv[1] == 'tuner':
        run_tuner()
    else:
        run_simulation() 