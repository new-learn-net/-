import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# 读取数据
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        x = df['x'].values
        y = df['y'].values
        return x, y
    except Exception as e:
        print(f"读取数据出错: {e}")
        # 生成更可靠的示例数据
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 3 * x + 5 + np.random.normal(0, 2, 100)  # 明确的线性关系
        print("使用示例数据进行演示")
        return x, y


# 模型和损失函数
def predict(x, w, b):
    return w * x + b


def loss_function(x, y, w, b):
    y_pred = predict(x, w, b)
    return np.mean((y_pred - y) ** 2)


# 训练模型（修改历史记录部分）
def train_model(x, y, learning_rate=0.01, epochs=1000):
    w = 0.0
    b = 0.0
    n = len(x)

    # 初始化历史记录列表（确保为空列表但已初始化）
    w_history = []
    b_history = []
    loss_history = []

    # 确保至少记录初始状态
    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss_function(x, y, w, b))

    for epoch in range(epochs):
        y_pred = predict(x, w, b)
        dw = (2 / n) * np.sum((y_pred - y) * x)
        db = (2 / n) * np.sum(y_pred - y)

        w -= learning_rate * dw
        b -= learning_rate * db

        loss = loss_function(x, y, w, b)

        # 每5轮记录一次（增加记录频率）
        if epoch % 5 == 0:
            w_history.append(w)
            b_history.append(b)
            loss_history.append(loss)

        # 打印训练进度和历史记录长度（用于调试）
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")
            print(f"历史记录长度: w={len(w_history)}, b={len(b_history)}, loss={len(loss_history)}")

    # 训练结束前再记录一次最终状态
    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss)

    return w, b, w_history, b_history, loss_history


# 绘制关系图（增加数据检查）
def plot_relationships(w_history, b_history, loss_history):
    # 检查数据是否有效
    print(f"\n绘图数据检查:")
    print(f"w历史记录长度: {len(w_history)}, 示例数据: {w_history[:3]}...")
    print(f"b历史记录长度: {len(b_history)}, 示例数据: {b_history[:3]}...")
    print(f"loss历史记录长度: {len(loss_history)}, 示例数据: {loss_history[:3]}...")

    # 如果数据不足，提示并退出
    if len(w_history) < 2 or len(loss_history) < 2:
        print("错误: 没有足够的历史数据绘制关系图")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制w和loss的关系
    ax1.plot(w_history, loss_history, 'b-', linewidth=2, markersize=3)
    ax1.set_title('w与loss的关系')
    ax1.set_xlabel('w值')
    ax1.set_ylabel('损失值 (loss)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 绘制b和loss的关系
    ax2.plot(b_history, loss_history, 'r-', linewidth=2, markersize=3)
    ax2.set_title('b与loss的关系')
    ax2.set_xlabel('b值')
    ax2.set_ylabel('损失值 (loss)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# 绘制拟合曲线
def plot_fitted_line(x, y, w, b):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.6, label='原始数据')
    plt.plot(x, predict(x, w, b), 'r-', linewidth=2, label=f'拟合直线: y = {w:.4f}x + {b:.4f}')
    plt.title('线性回归拟合结果')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()


# 主函数
def main():
    x, y = load_data('train.csv')
    w, b, w_history, b_history, loss_history = train_model(x, y)
    print(f"\n训练完成! 最终参数: w = {w:.4f}, b = {b:.4f}")
    plot_relationships(w_history, b_history, loss_history)
    plot_fitted_line(x, y, w, b)


if __name__ == "__main__":
    main()
