# %% 库

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# %% 主要参数

# 网格大小
Δ = np.linspace(0.3, 1.1, 9)
# 松弛因子
ω = np.linspace(0.8, 1.9, 100)

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

# %% 迭代构造

# 思路：利用SOR

# SOR迭代与进度条创建
with tqdm(total = len(Δ)*len(ω), desc="Progress") as pbar:
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
                max_error = 0.0
                for i in range(1, Nx-1):
                    for j in range(1, Ny-1):
                        temp = (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1]) / 4
                        delta = omega * (temp - T[i, j])
                        T[i, j] += delta
                        max_error = max(max_error, abs(delta))
                if max_error < tol:
                    break
                
            # 停止计时
            t1 = time.time()
            
            # 计时
            times[h_idx, omega_idx] = t1 - t0 # 单位：s
            
            # 迭代步统计
            flags[h_idx, omega_idx] = step
            
            # 进度条更新
            pbar.update(1)
            

# %% 不同松弛因子的收敛速度（迭代步数）

# 记录迭代次数的记录
#flags = np.zeros((len(Δ), len(ω)))

plt.figure(figsize=(18, 10))

# 颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(Δ)))

for delta_idx, delta in enumerate(Δ):
    iterations = flags[delta_idx, :]
    
    # 绘制曲线
    plt.plot(ω, iterations, 'o-', 
             color = colors[delta_idx], 
             linewidth = 2,
             markersize = 6,
             label = f'Δ={delta:.2f}')

plt.xlabel('Relaxation Factor (ω)', fontsize = 12)
plt.ylabel('Iteration Steps', fontsize = 12)
plt.title('Convergence Steps vs Relaxation Factor', fontsize = 14)
plt.legend(title='Grid Size (Δ)', fontsize = 10, title_fontsize = 11, 
           loc = 'upper right')
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


# %% 收集最佳松弛因子

plt.figure(figsize=(18, 10))
opt_factor_steps = []
opt_factor_times = []

# 迭代步数意义下
for delta_idx, delta in enumerate(Δ):
    iterations = flags[delta_idx, :]
    min_iter = np.argmin(iterations)
    opt_omega1 = ω[min_iter]
    opt_factor_steps.append(opt_omega1)

# 迭代时间意义下
for delta_idx, delta in enumerate(Δ):
    iterations = flags[delta_idx, :]
    min_iter = np.argmin(iterations)
    opt_omega2 = ω[min_iter]
    opt_factor_times.append(opt_omega2)

opt_factors = [opt_factor_steps, opt_factor_times]

plt.figure(figsize=(18, 10))
fig, axs = plt.subplots(2, 1, figsize = (18, 10), squeeze = False)
axs = axs.ravel()
fig.suptitle('Optimal Relaxtion Factor Varying With Grid Size', fontsize = 16, y = 0.99)
titles = ['Optimal Relaxtion Factor (Iteraion Steps)', 'Optimal Relaxtion Factor (Iteraion Time)']  

for i in range(len(opt_factors)):
    ax = axs[i]
    opt_factor = opt_factors[i]
    ax.plot(Δ, opt_factor)
    ax.set(xlabel = 'Grid Size', ylabel = 'Relaxtion Factor')
    ax.set_title(titles[i], fontsize=11, pad=15)
    print('')
    print(titles[i])
    for j in opt_factor:
        print(f'Optimal ω = {j:.2f} | Grid Size = {Δ[opt_factor.index(j)]:.2f}')
    print('')
    
plt.subplots_adjust(
    top = 0.9, 
    hspace = 0.35, 
    wspace = 0.20)

plt.tight_layout()
plt.show()




