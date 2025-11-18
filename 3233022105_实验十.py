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
    learning_rate = 0.001  # RNN对学习率更敏感，降低学习率
    epochs = 10
    # RNN专用参数（参考PPT）
    input_size = 28  # 每个时间步的输入维度（像素数）
    seq_len = 28  # 时间步长度（图像高度）
    hidden_size = 128  # 隐藏层维度
    num_layers = 2  # RNN层数
    # 路径配置
    data_root = "../dataset/mnist/"
    # 设备配置（自动检测GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 可视化参数
    plt_style = "seaborn-v0_8-whitegrid"


# 应用配置
cfg = Config()
plt.style.use(cfg.plt_style)


# -------------------------- 数据加载（复用并适配RNN输入格式）--------------------------
def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    os.makedirs(cfg.data_root, exist_ok=True)

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

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        pin_memory=True
    )

    return train_loader, test_loader


# -------------------------- RNN模型定义（基于PPT实现）--------------------------
class RNNCellNet(nn.Module):
    """基于PPT RNNCell的手动循环实现"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义多层RNNCell（PPT中的单个递归单元堆叠）
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        # 输出层（映射隐藏层到类别）
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size) → 适配batch_first
        batch_size = x.size(0)
        # 初始化隐藏状态（num_layers, batch_size, hidden_size）
        hidden = [torch.zeros(batch_size, self.hidden_size).to(cfg.device) for _ in range(self.num_layers)]

        # 手动循环每个时间步（PPT核心逻辑）
        for t in range(cfg.seq_len):
            x_t = x[:, t, :]  # 取第t个时间步的输入 (batch_size, input_size)
            for i, cell in enumerate(self.rnn_cells):
                hidden[i] = cell(x_t, hidden[i])
                x_t = hidden[i]  # 作为下一层的输入

        # 取最后一层最后一个时间步的隐藏状态作为输出
        out = self.fc(hidden[-1])
        return out


class RNNNet(nn.Module):
    """基于PPT nn.RNN的封装实现（自动处理时序）"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_first=True):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=0.5  # 增加dropout防止过拟合
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # 前向传播得到输出和最终隐藏状态
        out, hidden = self.rnn(x)  # out shape: (batch_size, seq_len, hidden_size)
        # 取最后一个时间步的输出用于分类
        out = self.fc(out[:, -1, :])  # (batch_size, num_classes)
        return out


# -------------------------- 工具函数（复用原有逻辑）--------------------------
def evaluate_model(model, test_loader, device=cfg.device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            # 调整输入格式为RNN所需：(batch_size, seq_len, input_size)
            inputs = inputs.squeeze(1).to(device)  # (batch_size, 28, 28)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device=cfg.device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 调整输入格式
        inputs = inputs.squeeze(1).to(device)  # (batch_size, 28, 28)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()

        if batch_idx % 300 == 299:
            avg_loss = running_loss / 300
            print(f"Epoch: {epoch + 1:2d}, Batch: {batch_idx + 1:4d}, Loss: {avg_loss:.3f}")
            running_loss = 0.0

    train_acc = 100. * correct_train / total_train
    avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    return train_acc, avg_loss


# -------------------------- 训练流程（复用并适配RNN）--------------------------
def train_model(model_name, model_class, train_loader, test_loader):
    print("\n" + "=" * 50)
    print(f"Training {model_name}...")
    print("=" * 50)

    # 初始化模型（RNN参数传入）
    if model_name == "RNNCell Network":
        model = model_class(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=10
        ).to(cfg.device)
    else:
        model = model_class(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=10
        ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)  # RNN用Adam更稳定

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    start_time = time.time()

    for epoch in range(cfg.epochs):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)

        print(f"{model_name} - Epoch: {epoch + 1:2d}")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print("-" * 40)

    training_time = time.time() - start_time

    return {
        "model_name": model_name,
        "train_acc": train_accuracies,
        "test_acc": test_accuracies,
        "train_loss": train_losses,
        "time": training_time,
        "model": model,
        "params": sum(p.numel() for p in model.parameters())
    }


# -------------------------- 可视化（复用原有逻辑）--------------------------
def plot_single_results(results):
    model_name = results["model_name"]
    epochs_range = range(1, cfg.epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{model_name} - Training Results", fontsize=16, fontweight="bold")

    axes[0].plot(epochs_range, results["train_acc"], "b-o", label="Train Accuracy", linewidth=2, markersize=6)
    axes[0].plot(epochs_range, results["test_acc"], "r-s", label="Test Accuracy", linewidth=2, markersize=6)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)

    axes[1].plot(epochs_range, results["train_loss"], "g-^", label="Train Loss", linewidth=2, markersize=6)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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


def plot_comparison_results(rnncell_results, rnn_results):
    epochs_range = range(1, cfg.epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RNNCell vs RNN Network Comparison", fontsize=16, fontweight="bold")

    axes[0, 0].plot(epochs_range, rnncell_results["train_acc"], "b-o", label="RNNCell Train", linewidth=2, markersize=6)
    axes[0, 0].plot(epochs_range, rnn_results["train_acc"], "r-o", label="RNN Train", linewidth=2, markersize=6)
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("Training Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 100)

    axes[0, 1].plot(epochs_range, rnncell_results["test_acc"], "b--s", label="RNNCell Test", linewidth=2, markersize=6)
    axes[0, 1].plot(epochs_range, rnn_results["test_acc"], "r--s", label="RNN Test", linewidth=2, markersize=6)
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)

    models = ["RNNCell", "RNN"]
    times = [rnncell_results["time"], rnn_results["time"]]
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

    x = np.arange(len(models))
    width = 0.35
    accuracies = [rnncell_results["test_acc"][-1], rnn_results["test_acc"][-1]]
    params = [rnncell_results["params"] / 1e4, rnn_results["params"] / 1e4]

    axes2 = axes[1, 1].twinx()
    bars2 = axes[1, 1].bar(x - width / 2, accuracies, width, label="Final Test Acc", color="green", alpha=0.7)
    bars3 = axes2.bar(x + width / 2, params, width, label="Params (10k)", color="orange", alpha=0.7)

    axes[1, 1].set_xlabel("Model")
    axes[1, 1].set_ylabel("Accuracy (%)", color="green")
    axes2.set_ylabel("Parameters (10k)", color="orange")
    axes[1, 1].set_title("Accuracy vs Parameters")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].set_ylim(0, 100)

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


# -------------------------- 主函数 --------------------------
def main():
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders()
    print(f"Dataset loaded successfully (Device: {cfg.device})")

    # 训练两个RNN模型
    rnncell_results = train_model("RNNCell Network", RNNCellNet, train_loader, test_loader)
    rnn_results = train_model("RNN Network", RNNNet, train_loader, test_loader)

    # 打印对比结果
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON RESULTS")
    print("=" * 60)
    for res in [rnncell_results, rnn_results]:
        print(f"\n{res['model_name']}:")
        print(f"  Final Train Accuracy: {res['train_acc'][-1]:.2f}%")
        print(f"  Final Test Accuracy: {res['test_acc'][-1]:.2f}%")
        print(f"  Training Time: {res['time']:.2f}s")
        print(f"  Parameters: {res['params']:,}")

    # 可视化
    print("\nPlotting single model results...")
    plot_single_results(rnncell_results)
    plot_single_results(rnn_results)

    print("Plotting comparison results...")
    plot_comparison_results(rnncell_results, rnn_results)


if __name__ == "__main__":
    main()