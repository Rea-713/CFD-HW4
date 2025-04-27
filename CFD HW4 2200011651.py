# %% 模块

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import itertools
from tqdm import tqdm

# %% 思路：由笔记的松弛迭代一节，案例2:Laplace / Poission方程的构造

# 因为流场内无热源，热传导系数为常数，温度T满足Laplace方程：∂²T/∂x² + ∂²T/∂y² = 0
# 使用有限差分法，构造中心差分格式离散方程
# (T[i+1,j] + T[i-1,j] -2*T[i,j]) / Δx² + (T[i,j+1] + T[i,j-1] -2*T[i,j]) / Δy² = 0
# 画成均匀网格，使得 Δx = Δy = Δ
# 带入得：T[i,j] =  (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]） / 4

# %% 参数

# 网格大小
Δ = [0.05, 0.3, 0.8, 1]

# x方向网格数
# +1的原因:创造边界条件
def create_Nx(h):
    Nx = int(15 / h) + 1 
    return Nx 

#y方向网格数
def create_Ny(h):
    Ny = int(12 / h) + 1 
    return Ny 

# 初始化温度场（Nx * Ny的矩阵）
# T = np.zeros((Nx, Ny))  

# 左、右、上、下侧边界
# T[:, -1], T[:, 0], T[0, :], T[-1, :]  = 20, 20, 100, 20


# %% 迭代构造

# 思路：利用Jacobi Over-Relaxaion

# 不同的松弛因子
ω = [0.2, 0.5, 0.7, 1]

# 迭代步数
iter_steps = int(1e6)

# 精度
tol = 1e-5

# 记录迭代次数
flags = np.zeros((len(Δ), len(ω)))

# 记录迭代时间
times = np.zeros((len(Δ), len(ω)))

# 颜色条刻度
GLOBAL_VMIN = 0
GLOBAL_VMAX = 100
TICK_INTERVAL = 10

# 加权Jacobi迭代与进度条创建
with tqdm(total=len(Δ)*len(ω), desc="Progress") as pbar:
    # 循环网格长度
    for h_idx, h in enumerate(Δ):
        
        # 创建画布
        plt.figure(figsize=(18, 10))

        # 动态绘图
        n_tu = len(Δ)
        rows = int(np.ceil(n_tu**0.5))  
        cols = int(np.ceil(n_tu / rows)) if rows > 0 else 1
        
        # 创建子图
        fig, axs = plt.subplots(rows, cols, figsize = (18, 8), squeeze = False)
        fig.suptitle(f'Temperature Distribution (Δ={h})', fontsize = 16, y = 0.95)
        axs = axs.ravel()

        # 移除多余子图
        for ax in axs[n_tu:]:
            ax.remove()
        
        # 循环松弛因子
        for omega_idx, omega in enumerate(ω):
            ax = axs[omega_idx]
            Nx = create_Nx(h)
            Ny = create_Ny(h)
            T = np.zeros((Nx, Ny)) 
            T[:, -1], T[:, 0], T[0, :], T[-1, :]  = 20, 20, 100, 20 #左、右、上、下边界条件
            t0 = time.time() # 开始计时
            for step in range(iter_steps):
                max_error = 0
                T_new = np.copy(T)  # 复制当前温度场
                for i in range(1, Nx-1):
                    for j in range(1, Ny-1):
                        T_new[i, j] = (1 - omega) * T[i, j] + omega * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]) / 4 
                        max_error = max(max_error, abs(T_new[i,j] - T[i,j]))
                
                # 误差分析，检查收敛
                if max_error < tol:
                    flags[h_idx, omega_idx] = step
                    converged = True
                    break
                
                # 最终的温度场
                T = np.copy(T_new)     
                
            if not converged:
                flags[h_idx, omega_idx] = iter_steps
                
            # 停止计时
            t1 = time.time()
            
            # 计时
            times[h_idx, omega_idx] = t1 - t0 # 单位：s
            
            # 进度条更新
            pbar.update(1)
            
            # 绘制云图
            im = ax.contourf(T, 
                      levels = 80,
                      origin = 'upper',
                      extent=[0, 15, 0, 12],
                      cmap = 'plasma')
            
            # 云图绘图设置
            ax.set(xlabel = 'X (cm)', 
                   ylabel = 'Y (cm)', 
                   title = f'ω={omega}, Steps={flags[h_idx, omega_idx]:.0f}')
            
            # 设置坐标轴字号大小
            ax.tick_params(axis = 'both', which = 'major', labelsize = 8) 
            
            # 取消网格
            ax.grid(False)
    
            # 刻度颜色条设置
            cbar_ticks = np.arange(GLOBAL_VMIN, GLOBAL_VMAX+TICK_INTERVAL, TICK_INTERVAL)
            
            # 创建颜色条
            cbar = plt.colorbar(im, ax = ax, pad = 0.02, ticks=cbar_ticks)
            
            # 颜色条标注字号大小
            cbar.ax.tick_params(labelsize = 8)
            
            # 颜色条标注
            cbar.set_label('Temperature (°C)', fontsize=10)
        
        # 子图位置调整
        plt.subplots_adjust(
            top = 0.85, 
            hspace = 0.35, 
            wspace = 0.20)
            
# %%

plt.show()
