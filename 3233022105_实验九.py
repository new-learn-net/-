import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 设备配置与全局参数 =====================
# 设置设备（GPU优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 全局参数
batch_size = 64
epochs = 10
lr = 0.001


# ===================== 2. 数据加载与预处理 =====================
def load_mnist_data():
    """加载并预处理MNIST数据集"""
    # 数据预处理：归一化+转Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值/标准差
    ])

    # 加载数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ===================== 3. GoogleNet（Inception模块） =====================
class InceptionA(nn.Module):
    """Inception模块（适配MNIST的轻量级版本）"""

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 1x1卷积分支
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 5x5卷积分支（1x1降维 + 5x5卷积）
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # 3x3卷积分支（1x1降维 + 两个3x3卷积）
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # 平均池化分支
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 按通道拼接
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class GoogleNet(nn.Module):
    """GoogleNet主网络"""

    def __init__(self):
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # 88=16+24+24+24

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)  # 适配MNIST输出维度

    def forward(self, x):
        in_size = x.size(0)

        # 第一层卷积+池化+Inception
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)

        # 第二层卷积+池化+Inception
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)

        # 展平+全连接
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


# ===================== 4. ResNet（残差模块） =====================
class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)  # 残差连接


class ResNet(nn.Module):
    """ResNet主网络"""

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = nn.Linear(512, 10)  # 适配MNIST输出维度

    def forward(self, x):
        in_size = x.size(0)

        # 第一层卷积+池化+残差块
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)

        # 第二层卷积+池化+残差块
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)

        # 展平+全连接
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


# ===================== 5. 训练与测试函数 =====================
def train(model, train_loader, criterion, optimizer, epoch):
    """单轮训练函数"""
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每300个batch打印日志
        if batch_idx % 300 == 299:
            print(f'[Epoch {epoch:2d}, Batch {batch_idx + 1:4d}] Loss: {running_loss / 300:.4f}')
            running_loss = 0.0


def test(model, test_loader):
    """测试函数（返回准确率）"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100 * correct / total
    print(f'测试集准确率: {acc:.2f}% ({correct}/{total})')
    return acc


# ===================== 6. 主函数（完整流程） =====================
def main():
    # 1. 加载数据
    train_loader, test_loader = load_mnist_data()
    print("数据加载完成！")

    # 2. 初始化模型、损失函数、优化器
    # GoogleNet初始化
    googlenet = GoogleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_gn = torch.optim.Adam(googlenet.parameters(), lr=lr)

    # ResNet初始化
    resnet = ResNet().to(device)
    optimizer_rn = torch.optim.Adam(resnet.parameters(), lr=lr)

    # 3. 训练GoogleNet
    print("\n" + "=" * 50)
    print("开始训练GoogleNet")
    print("=" * 50)
    gn_accs = []
    for epoch in range(1, epochs + 1):
        train(googlenet, train_loader, criterion, optimizer_gn, epoch)
        acc = test(googlenet, test_loader)
        gn_accs.append(acc)

    # 4. 训练ResNet
    print("\n" + "=" * 50)
    print("开始训练ResNet")
    print("=" * 50)
    rn_accs = []
    for epoch in range(1, epochs + 1):
        train(resnet, train_loader, criterion, optimizer_rn, epoch)
        acc = test(resnet, test_loader)
        rn_accs.append(acc)

    # 5. 结果可视化
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), gn_accs, label='GoogleNet', marker='o', linewidth=2)
    plt.plot(range(1, epochs + 1), rn_accs, label='ResNet', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('GoogleNet vs ResNet on MNIST', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))
    plt.ylim(90, 100)  # 聚焦准确率区间
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. 保存模型
    torch.save(googlenet.state_dict(), 'googlenet_mnist.pth')
    torch.save(resnet.state_dict(), 'resnet_mnist.pth')
    print("\n模型已保存：googlenet_mnist.pth / resnet_mnist.pth")
    print("准确率对比图已保存：accuracy_comparison.png")


# ===================== 运行主函数 =====================
if __name__ == "__main__":
    main()