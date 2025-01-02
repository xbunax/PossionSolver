#!/opt/homebrew/Caskroom/miniconda/base/envs/vtk/bin/python

import pyvista as pv
import numpy as np
import sys
import matplotlib.pyplot as plt

vtk_file = sys.argv[1]  # 替换为实际路径
grid = pv.read(vtk_file)

# 创建绘图对象
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="plasma", show_edges=True)
plotter.add_scalar_bar(title="Solution")

# 设置相机视角
plotter.camera_position = "xy"

# # 显示结果
target_y = 0.5  # 替换为你想筛选的 y 坐标值
tolerance = 0.01  # 容忍范围
y_values = grid.points[:, 1]  # 获取所有点的 y 坐标
mask = (y_values >= target_y - tolerance) & (y_values <= target_y + tolerance)
filtered_points = grid.points[mask]

# 获取标量值
if len(grid.point_data.keys()) > 0:
    scalar_field = grid.point_data[grid.point_data.keys()[0]]
    filtered_scalars = scalar_field[mask]

    # 根据 x 坐标排序并绘图
    x_values = filtered_points[:, 0]
    sorted_indices = np.argsort(x_values)
    x_sorted = x_values[sorted_indices]
    scalar_sorted = filtered_scalars[sorted_indices]

    plt.plot(x_sorted, scalar_sorted, marker="o", linestyle="-")
    plt.title("Scalar Data Along a Row (Target Y)")
    plt.xlabel("X-axis")
    plt.ylabel("Scalar Value")
    plt.grid(True)
    plt.show()
plotter.show()
