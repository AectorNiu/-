import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from typing import List, Tuple

class MultiFrameOptimizer:
    def __init__(self):
        # 初始化参数
        self.setup_parameters()
        
    def setup_parameters(self):
        """设置系统参数"""
        # 运动参数
        self.T = 10.0       # 总时长(s)
        self.dt = 0.1       # 时间步长(s)
        self.true_vel = np.array([50, 350])  # 真目标速度(m/s)
        self.delta_r = 100  # 预设拖引量(m)
        
        # 误差参数
        self.sigma_theta = np.deg2rad(1)  # 角度误差(rad)
        self.sigma_d = 5.0                # 距离误差(m)
        
        # 优化参数
        self.min_window = 5        # 增大最小窗口大小
        self.max_window = 9        # 增大最大窗口大小
        self.adaptive_window = False # 禁用自适应窗口
        
    def generate_true_trajectory(self) -> np.ndarray:
        """生成真目标轨迹"""
        n = int(self.T / self.dt)
        return np.array([self.true_vel * t * self.dt for t in range(n)])
        
    
    def add_measurement_error(self, P_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """添加雷达测量误差"""
        # 极坐标转换
        r = np.linalg.norm(P_true, axis=1)
        theta = np.arctan2(P_true[:,1], P_true[:,0])
        
        # 添加误差
        r_err = r + np.random.normal(0, self.sigma_d, len(r))
        theta_err = theta + np.random.normal(0, self.sigma_theta, len(theta))
        
        # 转回直角坐标
        x = r_err * np.cos(theta_err)
        y = r_err * np.sin(theta_err)
        return np.column_stack((x, y)), np.column_stack((r_err, theta_err))
    
    def preset_fake_trajectory(self, P_measured: np.ndarray) -> np.ndarray:
        """生成预设假目标轨迹"""
        # 简单径向拖引
        return P_measured + np.array([self.delta_r, 0])
    
    def multi_frame_optimization(self, P_preset: np.ndarray) -> np.ndarray:
        """多帧联合优化"""
        n = len(P_preset)
        P_opt = np.zeros_like(P_preset)
        
        for i in range(n):
            window_size = self.max_window
            start = max(0, i - window_size//2)
            end = min(n, i + window_size//2 + 1)
            window_indices = range(start, end)
            
            # 仅优化位置偏差
            def position_objective(x, P_preset, _):
                return 100.0 * np.sum((x.reshape(-1,2) - P_preset)**2)
                
            res = minimize(
                position_objective,
                P_preset[window_indices].flatten(),
                args=(P_preset[window_indices], window_indices),
                method='SLSQP',
                constraints=self.get_constraints(P_preset[window_indices]),
                options={'maxiter': 3000, 'ftol': 1e-7}
            )
            
            if i - start == end - 1 - i:
                P_opt[i] = res.x.reshape(-1,2)[i - start]
            else:
                P_opt[i] = P_preset[i]
                
        return P_opt
    
    def multi_frame_objective(self, x: np.ndarray, P_preset: np.ndarray, 
                            indices: List[int]) -> float:
        """多帧优化目标函数"""
        P_opt = x.reshape(-1,2)
        
        # 直接最小化位置偏差
        return 100.0 * np.sum((P_opt - P_preset)**2)
    
    def get_constraints(self, P_preset: np.ndarray):
        """获取简化约束条件"""
        constraints = []
        
        # 位置偏移约束
        def position_constraint(x, i, P=P_preset):
            return 50 - np.linalg.norm(x.reshape(-1,2)[i] - P[i])  # 最大偏移50m
        
        for i in range(len(P_preset)):
            constraints.append({'type': 'ineq', 'fun': position_constraint, 'args': (i,)})
            
        return constraints
    
    def evaluate_performance(self, P_true: np.ndarray, P_preset: np.ndarray, 
                           P_optimized: np.ndarray):
        """评估性能指标"""
        # 误差计算
        errors_preset = np.linalg.norm(P_preset - P_true, axis=1)
        errors_opt = np.linalg.norm(P_optimized - P_true, axis=1)
        
        # 速度分析
        v_preset = np.linalg.norm(np.diff(P_preset, axis=0)/self.dt, axis=1)
        v_opt = np.linalg.norm(np.diff(P_optimized, axis=0)/self.dt, axis=1)
        true_speed = np.linalg.norm(self.true_vel)
        
        # 平滑度分析
        def calculate_smoothness(traj):
            velocities = np.diff(traj, axis=0)/self.dt
            accels = np.diff(velocities, axis=0)/self.dt
            return np.mean(np.linalg.norm(accels, axis=1))
            
        smoothness_preset = calculate_smoothness(P_preset)
        smoothness_opt = calculate_smoothness(P_optimized)
        
        # 打印性能指标
        print("\n=== 性能评估 ===")
        print(f"预设轨迹平均误差: {np.mean(errors_preset):.2f}m")
        print(f"优化轨迹平均误差: {np.mean(errors_opt):.2f}m")
        print(f"误差降低比例: {(np.mean(errors_preset)-np.mean(errors_opt))/np.mean(errors_preset)*100:.1f}%")
        print(f"速度稳定性提升: {np.std(v_preset)-np.std(v_opt):.2f}m/s")
        print(f"轨迹平滑度提升: {smoothness_preset-smoothness_opt:.2f}m/s²")
        
        # 可视化
        self.visualize_results(P_true, P_preset, P_optimized, 
                              errors_preset, errors_opt, 
                              v_preset, v_opt, true_speed,
                              smoothness_preset, smoothness_opt)
    
    def visualize_results(self, P_true: np.ndarray, P_preset: np.ndarray, 
                         P_optimized: np.ndarray, errors_preset: np.ndarray, 
                         errors_opt: np.ndarray, v_preset: np.ndarray, 
                         v_opt: np.ndarray, true_speed: float,
                         smoothness_preset: float, smoothness_opt: float):
        """可视化分析结果"""
        plt.style.use('default')
        plt.figure(figsize=(18, 12))
        
        # 轨迹对比
        plt.subplot(2, 3, 1)
        plt.plot(P_true[:,0], P_true[:,1], 'g-', label='True')
        plt.plot(P_preset[:,0], P_preset[:,1], 'r--', label='Preset')
        plt.plot(P_optimized[:,0], P_optimized[:,1], 'b-.', label='Optimized')
        plt.legend()
        plt.title('Trajectory Comparison')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        # 误差对比
        plt.subplot(2, 3, 2)
        plt.plot(errors_preset, 'r--', label='Preset')
        plt.plot(errors_opt, 'b-', label='Optimized')
        plt.legend()
        plt.title('Position Error Comparison')
        plt.xlabel('Frame')
        plt.ylabel('Error (m)')
        
        # 速度分析
        plt.subplot(2, 3, 3)
        plt.plot(v_preset, 'r--', label='Preset')
        plt.plot(v_opt, 'b-', label='Optimized')
        plt.axhline(true_speed, color='g', linestyle=':', label='True')
        plt.legend()
        plt.title('Speed Analysis')
        plt.xlabel('Frame')
        plt.ylabel('Speed (m/s)')
        
        # 误差分布
        plt.subplot(2, 3, 4)
        plt.hist(errors_preset, bins=20, alpha=0.5, label='Preset')
        plt.hist(errors_opt, bins=20, alpha=0.5, label='Optimized')
        plt.legend()
        plt.title('Error Distribution')
        plt.xlabel('Error (m)')
        plt.ylabel('Frequency')
        
        # 平滑度对比
        plt.subplot(2, 3, 5)
        plt.bar(['Preset', 'Optimized'], 
                [smoothness_preset, smoothness_opt],
                color=['red', 'blue'])
        plt.title('Smoothness Comparison')
        plt.ylabel('Avg Acceleration (m/s²)')
        
        # 加速度分析
        plt.subplot(2, 3, 6)
        a_preset = np.diff(v_preset)/self.dt
        a_opt = np.diff(v_opt)/self.dt
        plt.plot(a_preset, 'r--', label='Preset')
        plt.plot(a_opt, 'b-', label='Optimized')
        plt.legend()
        plt.title('Acceleration Analysis')
        plt.xlabel('Frame')
        plt.ylabel('Acceleration (m/s²)')
        
        plt.tight_layout()
        plt.savefig('multi_frame_optimization.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    optimizer = MultiFrameOptimizer()
    
    # 生成轨迹
    P_true = optimizer.generate_true_trajectory()
    P_measured, _ = optimizer.add_measurement_error(P_true)
    P_preset = optimizer.preset_fake_trajectory(P_measured)
    
    # 优化轨迹
    P_optimized = optimizer.multi_frame_optimization(P_preset)
    
    # 评估性能
    optimizer.evaluate_performance(P_true, P_preset, P_optimized)