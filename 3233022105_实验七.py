import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np
import os


# -------------------------- 配置项集中管理（核心优化）--------------------------
class Config:
    """超参数与路径配置，统一管理便于修改"""
    # 训练参数
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5
    epochs = 10
    # 路径配置
    data_root = "../dataset/mnist/"
    # 设备配置（自动检测GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 可视化参数
    plt_style = "seaborn-v0_8-whitegrid"  # 更美观的绘图风格


# 应用配置
cfg = Config()
plt.style.use(cfg.plt_style)


# -------------------------- 数据加载优化 --------------------------
def get_data_loaders():
    """封装数据加载逻辑，复用性更强"""
    # 数据预处理（保持原有逻辑，补充注释）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为张量并归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集统计均值/标准差
    ])

    # 确保数据集目录存在
    os.makedirs(cfg.data_root, exist_ok=True)

    # 加载数据集
    train_dataset = datasets.MNIST(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=cfg.data_root,
        train=False,
        download=True,
        transform=transform
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        pin_memory=True  # 加速GPU数据传输
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        pin_memory=True
    )

    return train_loader, test_loader


# -------------------------- 模型定义优化 --------------------------
class FCNet(nn.Module):
    """全连接网络（补充文档字符串，优化层命名）"""

    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128, 64], output_dim=10, dropout=0.5):
        super().__init__()  # 简化Python3语法
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 784)  # 展平图像（28x28→784）
        return self.model(x)


class CNNNet(nn.Module):
    """卷积神经网络（补充文档字符串，简化forward逻辑）"""

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # 展平（保留batch维度）
        x = self.fc_layers(x)
        return x


# -------------------------- 工具函数优化 --------------------------
def evaluate_model(model, test_loader, device=cfg.device):
    """评估模型（迁移到指定设备，优化统计逻辑）"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算，加速推理
        for inputs, targets in test_loader:
            # 数据迁移到设备
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device=cfg.device):
    """单轮训练封装（复用训练逻辑，减少冗余）"""
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 数据迁移到设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 梯度清零→前向传播→计算损失→反向传播→参数更新
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()

        # 每300个batch打印一次中间结果
        if batch_idx % 300 == 299:
            avg_loss = running_loss / 300
            print(f"Epoch: {epoch + 1:2d}, Batch: {batch_idx + 1:4d}, Loss: {avg_loss:.3f}")
            running_loss = 0.0

    # 计算本轮训练准确率和平均损失
    train_acc = 100. * correct_train / total_train
    avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    return train_acc, avg_loss


# -------------------------- 训练流程优化 --------------------------
def train_model(model_name, model_class, train_loader, test_loader):
    """统一训练流程（支持不同模型，减少代码冗余）"""
    print("\n" + "=" * 50)
    print(f"Training {model_name}...")
    print("=" * 50)

    # 初始化模型、损失函数、优化器（迁移到指定设备）
    model = model_class().to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum
    )

    # 记录训练过程
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    start_time = time.time()

    # 多轮训练
    for epoch in range(cfg.epochs):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        # 记录训练指标
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        # 测试集评估
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)

        # 打印本轮结果
        print(f"{model_name} - Epoch: {epoch + 1:2d}")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print("-" * 40)

    # 计算总训练时间
    training_time = time.time() - start_time

    return {
        "model_name": model_name,
        "train_acc": train_accuracies,
        "test_acc": test_accuracies,
        "train_loss": train_losses,
        "time": training_time,
        "model": model,
        "params": sum(p.numel() for p in model.parameters())  # 提前计算参数量
    }


# -------------------------- 可视化优化 --------------------------
def plot_single_results(results):
    """绘制单个模型的训练结果（优化图表美观度）"""
    model_name = results["model_name"]
    epochs_range = range(1, cfg.epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{model_name} - Training Results", fontsize=16, fontweight="bold")

    # 1. 准确率曲线
    axes[0].plot(epochs_range, results["train_acc"], "b-o", label="Train Accuracy", linewidth=2, markersize=6)
    axes[0].plot(epochs_range, results["test_acc"], "r-s", label="Test Accuracy", linewidth=2, markersize=6)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)

    # 2. 损失曲线
    axes[1].plot(epochs_range, results["train_loss"], "g-^", label="Train Loss", linewidth=2, markersize=6)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 最终性能指标
    metrics = ["Final Train Acc", "Final Test Acc", "Training Time"]
    values = [
        results["train_acc"][-1],
        results["test_acc"][-1],
        results["time"]
    ]
    colors = ["#87CEEB", "#FFB6C1", "#98FB98"]
    bars = axes[2].bar(metrics, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1)
    axes[2].set_ylabel("Value")
    axes[2].set_title("Final Performance")

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label = f"{val:.2f}%" if height > 10 else f"{val:.2f}s"
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            height + (1 if height > 10 else 0.1),
            label,
            ha="center", va="bottom", fontweight="bold"
        )

    plt.tight_layout()
    plt.show()


def plot_comparison_results(fc_results, cnn_results):
    """对比两个模型的结果（优化子图布局）"""
    epochs_range = range(1, cfg.epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("FC vs CNN Network Comparison", fontsize=16, fontweight="bold")

    # 1. 训练准确率对比
    axes[0, 0].plot(epochs_range, fc_results["train_acc"], "b-o", label="FC Train", linewidth=2, markersize=6)
    axes[0, 0].plot(epochs_range, cnn_results["train_acc"], "r-o", label="CNN Train", linewidth=2, markersize=6)
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("Training Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 100)

    # 2. 测试准确率对比
    axes[0, 1].plot(epochs_range, fc_results["test_acc"], "b--s", label="FC Test", linewidth=2, markersize=6)
    axes[0, 1].plot(epochs_range, cnn_results["test_acc"], "r--s", label="CNN Test", linewidth=2, markersize=6)
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)

    # 3. 训练时间对比
    models = ["FC Network", "CNN Network"]
    times = [fc_results["time"], cnn_results["time"]]
    colors = ["blue", "red"]
    bars1 = axes[1, 0].bar(models, times, color=colors, alpha=0.7, edgecolor="black")
    axes[1, 0].set_ylabel("Time (seconds)")
    axes[1, 0].set_title("Training Time")
    for bar, val in zip(bars1, times):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.1,
            f"{val:.2f}s",
            ha="center", va="bottom", fontweight="bold"
        )

    # 4. 参数量与最终准确率对比（新增参数量维度）
    x = np.arange(len(models))
    width = 0.35
    accuracies = [fc_results["test_acc"][-1], cnn_results["test_acc"][-1]]
    params = [fc_results["params"] / 1e4, cnn_results["params"] / 1e4]  # 转为万为单位

    axes2 = axes[1, 1].twinx()  # 双Y轴
    bars2 = axes[1, 1].bar(x - width / 2, accuracies, width, label="Final Test Acc", color="green", alpha=0.7)
    bars3 = axes2.bar(x + width / 2, params, width, label="Params (10k)", color="orange", alpha=0.7)

    axes[1, 1].set_xlabel("Model")
    axes[1, 1].set_ylabel("Accuracy (%)", color="green")
    axes2.set_ylabel("Parameters (10k)", color="orange")
    axes[1, 1].set_title("Accuracy vs Parameters")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].set_ylim(0, 100)

    # 添加数值标签
    for bar, val in zip(bars2, accuracies):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.2f}%",
            ha="center", va="bottom", fontweight="bold"
        )
    for bar, val in zip(bars3, params):
        axes2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontweight="bold"
        )

    plt.tight_layout()
    plt.show()


# -------------------------- 主函数优化 --------------------------
def main():
    """主函数（简化逻辑，增强可读性）"""
    # 加载数据
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders()
    print(f"Dataset loaded successfully (Device: {cfg.device})")

    # 训练两个模型
    fc_results = train_model("FC Network", FCNet, train_loader, test_loader)
    cnn_results = train_model("CNN Network", CNNNet, train_loader, test_loader)

    # 打印详细对比结果
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON RESULTS")
    print("=" * 60)
    for res in [fc_results, cnn_results]:
        print(f"\n{res['model_name']}:")
        print(f"  Final Train Accuracy: {res['train_acc'][-1]:.2f}%")
        print(f"  Final Test Accuracy: {res['test_acc'][-1]:.2f}%")
        print(f"  Training Time: {res['time']:.2f}s")
        print(f"  Parameters: {res['params']:,}")

    # 绘制可视化结果
    print("\nPlotting single model results...")
    plot_single_results(fc_results)
    plot_single_results(cnn_results)

    print("Plotting comparison results...")
    plot_comparison_results(fc_results, cnn_results)


if __name__ == "__main__":
    main()