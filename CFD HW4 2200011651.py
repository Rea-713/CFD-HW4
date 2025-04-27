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
def Nx(h):
    Nx = 15 / h
    return Nx + 1

#y方向网格数
def Ny(h):
    Ny = 12 / h
    return Ny + 1

# 初始化温度场（Nx * Ny的矩阵）
# T = np.zeros((Nx, Ny))  

# 左、右、上、下侧边界
# T[0, :], T[-1,], T[:, -1], T[:, 0]  = 20, 20, 100, 20


# %% 迭代构造

# 思路：利用加速Gauss-Seidal迭代法：x(k+1) = x(k) + omega * Δx

# 不同的松弛因子
omega = [0.5, 0.8, 1, 2]

# 迭代步数
iter_steps = 1e5

# 精度
tol = 1e-5

# 迭代
flag = 0
while flag < iter_steps:
    max_error = 0
    for h in Δ:
        Nx = Nx(h)
        Ny = Ny(h)
        T = np.zeros((Nx, Ny)) 
        T[0, :], T[-1,], T[:, -1], T[:, 0]  = 20, 20, 100, 20
        for i in range(Nx - 1):
            for j in range(Ny - 1):
        





