# %% 库

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# %% 思路：由笔记的松弛迭代一节，案例2:Laplace / Poission方程的构造

# 因为流场内无热源，热传导系数为常数，温度 T 满足Laplace方程：∂²T/∂x² + ∂²T/∂y² = 0
# 使用有限差分法，构造中心差分格式离散方程
# (T[i+1,j] + T[i-1,j] -2*T[i,j]) / Δx² + (T[i,j+1] + T[i,j-1] -2*T[i,j]) / Δy² = 0
# 画成均匀网格，使得 Δx = Δy = Δ
# 带入得：T[i,j] =  (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]） / 4

# %% 主要参数

# 网格大小
Δ = np.linspace(0.1, 0.3, 10)
# 松弛因子
ω = np.linspace(0.7, 1, 10)


# %% 次要参数

# x方向网格数
# +1的原因:创造边界条件
def create_Nx(h):
    Nx = int(15 / h) + 1 
    return Nx 

#y方向网格数
def create_Ny(h):
    Ny = int(12 / h) + 1 
    return Ny 

# 迭代步数
iter_steps = int(1e6)

# 精度
tol = 1e-5

# 记录迭代次数
flags = np.zeros((len(Δ), len(ω)))

# 记录迭代时间
times = np.zeros((len(Δ), len(ω)))

# 初始化温度场（Nx * Ny的矩阵）
# T = np.zeros((Nx, Ny))  

# 左、右、上、下侧边界
# T[:, -1], T[:, 0], T[0, :], T[-1, :]  = 20, 20, 100, 20

# %% 迭代构造与温度场云图绘图

# 思路：利用Jacobi Over-Relaxaion

# 加权Jacobi迭代与进度条创建
with tqdm(total=len(Δ)*len(ω), desc="Progress") as pbar:
    # 循环网格长度
    for h_idx, h in enumerate(Δ): 
        # 循环松弛因子
        for omega_idx, omega in enumerate(ω):
            Nx = create_Nx(h)
            Ny = create_Ny(h)
            T = np.zeros((Nx, Ny)) 
            T[:, -1], T[:, 0], T[0, :], T[-1, :]  = 20, 20, 20, 100 #左、右、上、下边界条件
            t0 = time.time() # 开始计时
            for step in range(iter_steps):
                T_new = np.copy(T)  # 复制当前温度场
                T_new[1:-1, 1:-1] = (1 - omega) * T[1:-1, 1:-1] + omega * (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2]) / 4   # 向量化
                
                # 计算相对误差
                rel_error = np.max(np.abs(T_new[1:-1, 1:-1] - T[1:-1, 1:-1]) / (T_new[1:-1, 1:-1] + 1e-8))
                
                if rel_error < tol:
                    flags[h_idx, omega_idx] = step
                    converged = True
                    break
                
                T = T_new
                
            if not converged:
                flags[h_idx, omega_idx] = iter_steps
                
            # 停止计时
            t1 = time.time()
            
            # 计时
            times[h_idx, omega_idx] = t1 - t0 # 单位：s
            
            # 进度条更新
            pbar.update(1)
            

# %% 不同松弛因子的收敛速度（迭代次数）

# 记录迭代次数的记录
#flags = np.zeros((len(Δ), len(ω)))

plt.figure(figsize=(18, 10))

# 颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(Δ)))

for delta_idx, delta in enumerate(Δ):
    iterations = flags[delta_idx, :]
    valid_flag = iterations < iter_steps
    valid_omega = ω[valid_flag]
    valid_iter = iterations[valid_flag]
    
    # 绘制曲线
    plt.plot(valid_omega, valid_iter, 'o-', 
             color=colors[delta_idx], 
             linewidth=2,
             markersize=6,
             label=f'Δ={delta:.2f}')

plt.xlabel('Relaxation Factor (ω)', fontsize=12)
plt.ylabel('Iteration Steps', fontsize=12)
plt.title('Convergence Steps vs Relaxation Factor', fontsize=14)
plt.legend(title='Grid Size (Δ)', fontsize=10, title_fontsize=11, 
           loc='upper right')
plt.tight_layout()
plt.show()

# %% 不同松弛因子的收敛速度（迭代时间）

# 迭代时间的记录
#times = np.zeros((len(Δ), len(ω)))

plt.figure(figsize=(18, 10))
for delta_idx, delta in enumerate(Δ):
    time_data = times[delta_idx, :]
    valid_flag = flags[delta_idx, :] < iter_steps
    valid_omega = ω[valid_flag]
    valid_time = time_data[valid_flag]
    
    plt.plot(valid_omega, valid_time, 's-', 
             color=colors[delta_idx], 
             linewidth=2,
             markersize=6,
             label=f'Δ={delta:.2f}')

plt.xlabel('Relaxation Factor (ω)', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.title('Computation Time vs Relaxation Factor', fontsize=14)
plt.legend(title='Grid Size (Δ)', fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.show()


# %%


