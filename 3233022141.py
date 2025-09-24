import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 配置中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
rcParams['figure.dpi'] = 100  # 提高图表清晰度


def main():
    """线性回归模型训练与可视化主函数"""
    # 1. 数据加载与基本信息查看
    data = pd.read_csv("train.csv")
    x_data = data['x'].values  # 特征数据
    y_data = data['y'].values  # 目标数据

    # 打印数据基本统计信息
    print(f"数据基本信息：")
    print(f"样本量：{len(x_data)}")
    print(f"x 取值范围：[{x_data.min():.2f}, {x_data.max():.2f}]")
    print(f"y 取值范围：[{y_data.min():.2f}, {y_data.max():.2f}]\n")

    # 2. 模型与损失函数定义
    def forward(x, w, b):
        """前向传播：计算线性回归预测值 y_pred = w*x + b"""
        return w * x + b

    def mse_loss(x, y, w, b):
        """计算均方误差损失"""
        y_pred = forward(x, w, b)
        return np.mean((y_pred - y) ** 2)

    # 3. 模型训练（梯度下降）
    # 初始化参数与超参数
    w, b = 0.0, 0.0  # 权重与偏置初始值
    learning_rate = 0.01  # 学习率
    num_epochs = 2000  # 训练轮次
    loss_history = []  # 记录训练过程中的损失变化

    # 梯度下降迭代
    for epoch in range(num_epochs):
        # 计算预测误差
        error = forward(x_data, w, b) - y_data

        # 计算梯度并更新参数（MSE损失的导数）
        w -= learning_rate * 2 * np.mean(error * x_data)
        b -= learning_rate * 2 * np.mean(error)

        # 记录当前轮次的损失
        current_loss = mse_loss(x_data, y_data, w, b)
        loss_history.append(current_loss)

        # 每100轮打印一次训练进度（可选）
        # if (epoch + 1) % 100 == 0:
        #     print(f"轮次 [{epoch+1}/{num_epochs}], 损失: {current_loss:.4f}")

    # 打印训练结果
    print(f"训练完成：")
    print(f"最优权重 w: {w:.3f}")
    print(f"最优偏置 b: {b:.3f}")
    print(f"最终均方误差 MSE: {loss_history[-1]:.3f}\n")

    # 4. 参数与损失函数关系可视化
    # 生成参数搜索范围
    w_range = np.linspace(0, 4, 100)  # w的搜索范围
    b_range = np.linspace(-2, 4, 100)  # b的搜索范围

    # 计算固定b时，不同w对应的损失（用于找最优w）
    w_losses = np.array([mse_loss(x_data, y_data, wi, b) for wi in w_range])
    optimal_w = w_range[np.argmin(w_losses)]

    # 计算固定w时，不同b对应的损失（用于找最优b）
    b_losses = np.array([mse_loss(x_data, y_data, w, bi) for bi in b_range])
    optimal_b = b_range[np.argmin(b_losses)]

    # 绘制参数与损失关系图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：w与损失函数关系
    ax1.plot(w_range, w_losses, 'b-', linewidth=2, alpha=0.8)
    ax1.axvline(optimal_w, color='red', linestyle='--', linewidth=1.5,
                label=f'最优w = {optimal_w:.2f}')
    ax1.set_title("权重参数 w 与损失函数关系（固定 b）", fontsize=12)
    ax1.set_xlabel("w 值", fontsize=10)
    ax1.set_ylabel("MSE 损失", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, linestyle="--")

    # 右图：b与损失函数关系
    ax2.plot(b_range, b_losses, 'g-', linewidth=2, alpha=0.8)
    ax2.axvline(optimal_b, color='red', linestyle='--', linewidth=1.5,
                label=f'最优b = {optimal_b:.2f}')
    ax2.set_title("偏置参数 b 与损失函数关系（固定 w）", fontsize=12)
    ax2.set_xlabel("b 值", fontsize=10)
    ax2.set_ylabel("MSE 损失", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()

    # 5. 绘制拟合结果
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color="blue", alpha=0.6, s=30, label="训练数据")
    plt.plot(x_data, forward(x_data, w, b), "r-", linewidth=2.5,
             label=f"拟合直线: y = {w:.2f}x + {b:.2f}")
    plt.title("线性回归拟合结果", fontsize=12)
    plt.xlabel("特征 x", fontsize=10)
    plt.ylabel("目标值 y", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()