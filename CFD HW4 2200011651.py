# %% 模块

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import itertools

# %% 思路：由笔记的松弛迭代一节，案例2:Laplace / Poission方程的构造

# 因为流场内无热源，热传导系数为常数，温度T满足Laplace方程：∂²T/∂x² + ∂²T/∂y² = 0
# 使用有限差分法，构造中心差分格式离散方程
# (T[i+1,j] + T[i-1,j] -2*T[i,j]) / Δx² + (T[i,j+1] + T[i,j-1] -2*T[i,j]) / Δy² = 0
# 画成均匀网格，使得 Δx = Δy = Δ
# 带入得：T[i,j] =  (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]） / 4

# %% 参数

# 网格大小
Δ = [0.1, 0.2, 0.5]

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
# T[0, :], T[-1,], T[:, -1], T[:, 0]  = 20, 20, 100, 20


# %% 迭代构造

# 思路：利用Jacobi Over-Relaxaion

# 不同的松弛因子
ω = [0.3, 0.5, 0.8, 1, 1.5]

# 迭代步数
iter_steps = 1e6

# 精度
tol = 1e-5

# 记录迭代次数
flags = np.zeros((len(Δ), len(ω))) 

# 记录迭代时间
times = np.zeros((len(Δ), len(ω)))

# 加权Jacobi迭代
for h in Δ:
    for omega in ω:
        Nx = create_Nx(h)
        Ny = create_Ny(h)
        T = np.zeros((Nx, Ny)) 
        T[0, :], T[-1,], T[:, -1], T[:, 0]  = 20, 20, 100, 20
        t0 = time.time()
        h_index = Δ.index(h)
        omega_index = ω.index(omega)
        flag = flags[h_index, omega_index] 
        while flag < iter_steps:
            max_error = 0
            T_new = np.copy(T)  # 复制当前温度场
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    T_new[i, j] = (1 - omega) * T[i, j] + omega * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]) / 4 
                    max_error = max(max_error, abs(T_new[i,j] - T[i,j]))
                    flag += 1
                
        t1 = time.time()
        time[h_index, omega_index] = t1 - t0 # 单位：s



