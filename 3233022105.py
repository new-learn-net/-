import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== 解决中文显示问题 ====
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==== 读取数据 ====
data = pd.read_csv("train.csv")
data = data.dropna(subset=['y'])  # 删除 y 为空的行
x_data, y_data = data['x'].values, data['y'].values
print(f"数据量: {len(x_data)}, x范围: [{x_data.min():.2f}, {x_data.max():.2f}], y范围: [{y_data.min():.2f}, {y_data.max():.2f}]")

# ==== 模型定义 ====
forward = lambda x, w, b: w * x + b
mse_loss = lambda x, y, w, b: np.mean((forward(x, w, b) - y) ** 2)

# ==== 初始化参数 ====
w, b, lr, epochs = 0.0, 0.0, 0.00001, 2000
loss_history = []
w_history = []

# ==== 训练 ====
for epoch in range(epochs):
    error = forward(x_data, w, b) - y_data
    w -= lr * np.mean(error * x_data) * 2
    b -= lr * np.mean(error) * 2
    loss_history.append(mse_loss(x_data, y_data, w, b))
    w_history.append(w)

print(f"\n训练完成: w={w:.3f}, b={b:.3f}, MSE={loss_history[-1]:.3f}")

# ==== 绘制 2x2 子图 ====
w_range = np.linspace(0, 2, 100)
b_range = np.linspace(-2, 2, 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(w_range, [mse_loss(x_data, y_data, wi, b) for wi in w_range], 'b-', linewidth=2)
axes[0, 0].set_xlabel("权重参数 w", fontsize=12)
axes[0, 0].set_ylabel("MSE 损失", fontsize=12)
axes[0, 0].set_title("w 与损失函数关系 (固定 b)", fontsize=14)
axes[0, 0].grid(True, linestyle="--", alpha=0.6)

axes[0, 1].plot(b_range, [mse_loss(x_data, y_data, w, bi) for bi in b_range], 'g-', linewidth=2)
axes[0, 1].set_xlabel("偏置参数 b", fontsize=12)
axes[0, 1].set_ylabel("MSE 损失", fontsize=12)
axes[0, 1].set_title("b 与损失函数关系 (固定 w)", fontsize=14)
axes[0, 1].grid(True, linestyle="--", alpha=0.6)

axes[1, 0].plot(range(epochs), w_history, 'm-', linewidth=2)
axes[1, 0].set_xlabel("迭代次数", fontsize=12)
axes[1, 0].set_ylabel("权重 w", fontsize=12)
axes[1, 0].set_title("w 随迭代次数变化", fontsize=14)
axes[1, 0].grid(True, linestyle="--", alpha=0.6)

axes[1, 1].plot(range(epochs), loss_history, 'c-', linewidth=2)
axes[1, 1].set_xlabel("迭代次数", fontsize=12)
axes[1, 1].set_ylabel("MSE 损失", fontsize=12)
axes[1, 1].set_title("损失随迭代次数变化", fontsize=14)
axes[1, 1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()  # 第一个图显示完再画第二个图

# ==== 线性回归拟合结果可视化 ====
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color="blue", alpha=0.6, label="训练数据")
plt.plot(x_data, forward(x_data, w, b), "r-", linewidth=2.5, label=f"拟合直线: y={w:.2f}x+{b:.2f}")
plt.xlabel("特征 x", fontsize=12)
plt.ylabel("目标值 y", fontsize=12)
plt.title("线性回归拟合结果", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()