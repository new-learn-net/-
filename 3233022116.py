import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

data = pd.read_csv("train.csv")
x_data, y_data = data['x'].values, data['y'].values
print(f"数据量: {len(x_data)}, x范围: [{x_data.min():.2f}, {x_data.max():.2f}], y范围: [{y_data.min():.2f}, {y_data.max():.2f}]")

forward = lambda x, w, b: w * x + b
mse_loss = lambda x, y, w, b: np.mean((forward(x, w, b) - y) ** 2)

w, b, lr, epochs = 0.0, 0.0, 0.01, 2000
loss_history = []

for epoch in range(epochs):
    error = forward(x_data, w, b) - y_data
    w -= lr * np.mean(error * x_data) * 2
    b -= lr * np.mean(error) * 2
    loss_history.append(mse_loss(x_data, y_data, w, b))

print(f"\n训练完成: w={w:.3f}, b={b:.3f}, MSE={loss_history[-1]:.3f}")

w_range, b_range = np.linspace(0, 4, 100), np.linspace(-2, 4, 100)
optimal_w = w_range[np.argmin([mse_loss(x_data, y_data, wi, b) for wi in w_range])]
optimal_b = b_range[np.argmin([mse_loss(x_data, y_data, w, bi) for bi in b_range])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(w_range, [mse_loss(x_data, y_data, wi, b) for wi in w_range], 'b-', linewidth=2)
ax1.axvline(optimal_w, color='red', linestyle='--', label=f'最优w={optimal_w:.2f}')
ax1.set(xlabel="权重参数 w", ylabel="MSE损失", title="w 与损失函数关系 (固定 b)")
ax1.legend(), ax1.grid(alpha=0.3, linestyle="--")

ax2.plot(b_range, [mse_loss(x_data, y_data, w, bi) for bi in b_range], 'g-', linewidth=2)
ax2.axvline(optimal_b, color='red', linestyle='--', label=f'最优b={optimal_b:.2f}')
ax2.set(xlabel="偏置参数 b", ylabel="MSE损失", title="b 与损失函数关系 (固定 w)")
ax2.legend(), ax2.grid(alpha=0.3, linestyle="--")
plt.tight_layout(), plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color="blue", alpha=0.6, label="训练数据")
plt.plot(x_data, forward(x_data, w, b), "r-", linewidth=2.5, label=f"拟合直线: y={w:.2f}x+{b:.2f}")
plt.xlabel("特征 x"), plt.ylabel("目标值 y"), plt.title("线性回归拟合结果")
plt.legend(), plt.grid(alpha=0.3, linestyle="--"), plt.show()