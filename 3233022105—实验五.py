import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ==== 解决中文显示问题 ====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==== 读取数据 ====
data = pd.read_csv("train.csv")
data = data.dropna(subset=['y'])

# 数据标准化处理，防止梯度爆炸
x_raw = data['x'].values
y_raw = data['y'].values

# 数据标准化
x_mean, x_std = x_raw.mean(), x_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()

x_normalized = (x_raw - x_mean) / x_std
y_normalized = (y_raw - y_mean) / y_std

x_data = torch.tensor(x_normalized, dtype=torch.float32).view(-1, 1)
y_data = torch.tensor(y_normalized, dtype=torch.float32).view(-1, 1)

print(
    f"数据量: {len(x_data)}, x范围: [{x_data.min():.2f}, {x_data.max():.2f}], y范围: [{y_data.min():.2f}, {y_data.max():.2f}]")
print(f"原始数据 - x范围: [{x_raw.min():.2f}, {x_raw.max():.2f}], y范围: [{y_raw.min():.2f}, {y_raw.max():.2f}]")


# ==== 定义线性回归模型 ====
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# ==== 训练函数 ====
def train_model(optimizer_class, optimizer_name, lr=0.01, epochs=1000):
    # 初始化模型
    model = LinearModel()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=lr)

    # 记录训练过程
    loss_history = []
    w_history = []
    b_history = []

    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录参数
        current_loss = loss.item()
        loss_history.append(current_loss)
        w_history.append(model.linear.weight.item())
        b_history.append(model.linear.bias.item())

        # 检查NaN值
        if np.isnan(current_loss):
            print(f"警告: {optimizer_name} 在第 {epoch} 轮出现NaN损失")
            break

    # 最终参数
    w_final = model.linear.weight.item()
    b_final = model.linear.bias.item()
    final_loss = loss_history[-1] if not np.isnan(loss_history[-1]) else float('inf')

    print(f"{optimizer_name}: w={w_final:.4f}, b={b_final:.4f}, 最终损失={final_loss:.4f}")

    return {
        'model': model,
        'loss_history': loss_history,
        'w_history': w_history,
        'b_history': b_history,
        'w_final': w_final,
        'b_final': b_final,
        'final_loss': final_loss
    }


# ==== 使用不同的优化器进行训练 ====
optimizers = [
    (torch.optim.SGD, 'SGD'),
    (torch.optim.Adam, 'Adam'),
    (torch.optim.Adagrad, 'Adagrad'),
    (torch.optim.RMSprop, 'RMSprop')
]

results = {}
for optimizer_class, name in optimizers:
    try:
        results[name] = train_model(optimizer_class, name, lr=0.01, epochs=1000)
    except Exception as e:
        print(f"{name} 训练失败: {e}")
        continue

# 过滤掉包含NaN的结果
valid_results = {name: result for name, result in results.items()
                 if not np.isnan(result['final_loss']) and result['final_loss'] != float('inf')}

if not valid_results:
    print("所有优化器都训练失败，尝试调整学习率...")
    # 尝试更小的学习率
    for optimizer_class, name in optimizers:
        try:
            results[name] = train_model(optimizer_class, name, lr=0.001, epochs=1000)
            if not np.isnan(results[name]['final_loss']):
                valid_results[name] = results[name]
        except Exception as e:
            print(f"{name} 训练失败: {e}")
            continue

if not valid_results:
    print("所有优化器都失败，请检查数据或模型")
    exit()

# ==== 可视化结果 ====
# 1. 不同优化器的损失曲线对比
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
for name in valid_results:
    # 只显示前100个epoch，且过滤掉NaN值
    valid_losses = [loss for loss in valid_results[name]['loss_history'][:100] if not np.isnan(loss)]
    if valid_losses:
        plt.plot(range(len(valid_losses)), valid_losses, label=name)
plt.xlabel('迭代次数')
plt.ylabel('MSE损失')
plt.title('不同优化器的损失下降曲线(前100轮)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 2. 权重w的变化过程
plt.subplot(2, 2, 2)
for name in valid_results:
    # 过滤掉NaN值
    valid_weights = [w for w in valid_results[name]['w_history'][:100] if not np.isnan(w)]
    if valid_weights:
        plt.plot(range(len(valid_weights)), valid_weights, label=name)
plt.xlabel('迭代次数')
plt.ylabel('权重 w')
plt.title('权重w随迭代次数的变化(前100轮)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 3. 偏置b的变化过程
plt.subplot(2, 2, 3)
for name in valid_results:
    # 过滤掉NaN值
    valid_biases = [b for b in valid_results[name]['b_history'][:100] if not np.isnan(b)]
    if valid_biases:
        plt.plot(range(len(valid_biases)), valid_biases, label=name)
plt.xlabel('迭代次数')
plt.ylabel('偏置 b')
plt.title('偏置b随迭代次数的变化(前100轮)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 4. 最终拟合结果对比
plt.subplot(2, 2, 4)
x_plot_normalized = torch.linspace(x_data.min(), x_data.max(), 100).view(-1, 1)

# 绘制原始数据点（标准化后的）
plt.scatter(x_data.numpy(), y_data.numpy(), color='black', alpha=0.6, label='训练数据(标准化)')

# 绘制各优化器的拟合直线
for name in valid_results:
    model = valid_results[name]['model']
    y_plot_normalized = model(x_plot_normalized).detach().numpy()
    plt.plot(x_plot_normalized.numpy(), y_plot_normalized, linewidth=2, label=f'{name}拟合')

plt.xlabel('特征 x (标准化)')
plt.ylabel('目标值 y (标准化)')
plt.title('不同优化器的拟合结果对比(标准化数据)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ==== 学习率和epochs的影响分析 ====
# 测试不同学习率
learning_rates = [0.001, 0.01, 0.1]
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for lr in learning_rates:
    try:
        result = train_model(torch.optim.Adam, f'Adam lr={lr}', lr=lr, epochs=500)
        valid_losses = [loss for loss in result['loss_history'] if not np.isnan(loss)]
        if valid_losses:
            plt.plot(range(len(valid_losses)), valid_losses, label=f'lr={lr}')
    except Exception as e:
        print(f"学习率 {lr} 训练失败: {e}")

plt.xlabel('迭代次数')
plt.ylabel('MSE损失')
plt.title('不同学习率对训练的影响(Adam)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.yscale('log')

# 测试不同epochs
plt.subplot(1, 2, 2)
epochs_list = [100, 500, 1000, 2000]
final_losses = []
for epochs in epochs_list:
    try:
        result = train_model(torch.optim.Adam, f'Adam epochs={epochs}', lr=0.01, epochs=epochs)
        if not np.isnan(result['final_loss']):
            final_losses.append(result['final_loss'])
        else:
            final_losses.append(float('inf'))
    except Exception as e:
        print(f"epochs {epochs} 训练失败: {e}")
        final_losses.append(float('inf'))

if any(loss != float('inf') for loss in final_losses):
    plt.bar(range(len(epochs_list)), final_losses)
    plt.xticks(range(len(epochs_list)), [str(e) for e in epochs_list])
    plt.xlabel('训练轮数')
    plt.ylabel('最终损失')
    plt.title('不同训练轮数对最终损失的影响')
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ==== 输出最佳模型 ====
if valid_results:
    best_result = min(valid_results.values(), key=lambda x: x['final_loss'])
    best_optimizer_name = [name for name, result in valid_results.items()
                           if result['final_loss'] == best_result['final_loss']][0]

    print(f"\n最佳优化器: {best_optimizer_name}")
    print(f"最佳参数: w={best_result['w_final']:.4f}, b={best_result['b_final']:.4f}")
    print(f"最小损失: {best_result['final_loss']:.4f}")

    # ==== 反标准化，得到原始数据的参数 ====
    # 标准化后的模型: y_norm = w_norm * x_norm + b_norm
    # 其中: x_norm = (x - x_mean) / x_std, y_norm = (y - y_mean) / y_std
    # 代入得: (y - y_mean) / y_std = w_norm * (x - x_mean) / x_std + b_norm
    # 整理得: y = (w_norm * y_std / x_std) * x + (y_mean + b_norm * y_std - w_norm * y_std * x_mean / x_std)

    w_original = best_result['w_final'] * y_std / x_std
    b_original = y_mean + best_result['b_final'] * y_std - best_result['w_final'] * y_std * x_mean / x_std

    print(f"\n原始数据空间的参数:")
    print(f"w_original = {w_original:.4f}")
    print(f"b_original = {b_original:.4f}")
    print(f"原始数据拟合直线: y = {w_original:.4f} * x + {b_original:.4f}")

    # ==== 在原始数据空间绘制拟合结果 ====
    plt.figure(figsize=(10, 6))
    plt.scatter(x_raw, y_raw, color='blue', alpha=0.6, label='训练数据')

    # 生成拟合直线
    x_line = np.linspace(x_raw.min(), x_raw.max(), 100)
    y_line = w_original * x_line + b_original
    plt.plot(x_line, y_line, 'r-', linewidth=2,
             label=f'拟合直线: y={w_original:.4f}x+{b_original:.4f}')

    plt.xlabel('特征 x')
    plt.ylabel('目标值 y')
    plt.title('线性回归拟合结果(原始数据)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # ==== 使用最佳模型进行预测 ====
    best_model = best_result['model']

    # 如果要预测新的x值，需要先标准化
    x_test_original = 4.0
    x_test_normalized = (x_test_original - x_mean) / x_std
    x_test = torch.tensor([[x_test_normalized]], dtype=torch.float32)
    y_test_normalized = best_model(x_test)

    # 反标准化得到原始空间的预测值
    y_test_original = y_test_normalized.item() * y_std + y_mean

    print(f"\n预测结果:")
    print(f"输入 x = {x_test_original:.1f}")
    print(f"预测 y = {y_test_original:.4f}")
else:
    print("没有有效的训练结果")