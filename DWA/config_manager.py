import os
import yaml

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