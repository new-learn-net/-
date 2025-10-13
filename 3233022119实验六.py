import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==== 解决中文显示问题 ====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==== 1. 数据读取与预处理 ====
def load_and_preprocess_data(filepath):
    # 读取数据集
    data = pd.read_csv(filepath)
    print("数据集基本信息:")
    print(f"数据集形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    print("\n缺失值统计:")
    print(data.isnull().sum())

    # 目标变量与特征选择（根据实际数据集列名调整，此处适配常见生态足迹数据集）
    # 若实际列名不同，需替换target_col为真实生态足迹相关列名（如"Total Ecological Footprint"）
    target_col = "Ecological Footprint"
    if target_col not in data.columns:
        # 备用目标列（若原数据集列名不同，需手动修改为正确列名）
        target_col = "Carbon Footprint" if "Carbon Footprint" in data.columns else data.columns[-1]
    print(f"\n选择的目标变量列: {target_col}")

    # 筛选数值型特征，排除非数值列（国家名称、地区等）
    feature_cols = [col for col in data.columns
                    if col not in [target_col, "Country", "Region", "Code"]
                    and data[col].dtype in [np.float64, np.int64]]
    print(f"选择的特征列: {feature_cols}")

    # 处理缺失值：删除目标变量缺失行，特征缺失值用均值填充
    data = data.dropna(subset=[target_col])
    for col in feature_cols:
        if data[col].isnull().sum() > 0:
            fill_val = data[col].mean()
            data[col].fillna(fill_val, inplace=True)
            print(f"特征 {col} 缺失值用均值 {fill_val:.4f} 填充")

    # 提取特征和目标变量
    X = data[feature_cols].values
    y = data[target_col].values.reshape(-1, 1)  # 转为二维数组适配模型

    # 数据划分（8:2训练集-测试集）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 标准化处理（避免数据泄露，仅用训练集参数）
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_normalized = scaler_X.fit_transform(X_train)
    X_test_normalized = scaler_X.transform(X_test)
    y_train_normalized = scaler_y.fit_transform(y_train)
    y_test_normalized = scaler_y.transform(y_test)

    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_normalized, dtype=torch.float32)

    print(f"\n数据预处理完成:")
    print(f"训练集: 特征形状 {X_train_tensor.shape}, 目标形状 {y_train_tensor.shape}")
    print(f"测试集: 特征形状 {X_test_tensor.shape}, 目标形状 {y_test_tensor.shape}")
    print(f"特征标准化均值: {scaler_X.mean_[:3]}...（仅显示前3个）")
    print(f"目标标准化均值: {scaler_y.mean_[0]:.4f}")

    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
            scaler_X, scaler_y, feature_cols, target_col)


# ==== 2. 自定义Dataset类 ====
class CountryFootprintDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        # 索引获取单个样本
        return self.X[index], self.y[index]

    def __len__(self):
        # 返回数据集总样本数
        return self.len


# ==== 3. 构建5层全连接神经网络 ====
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim):
        super(FullyConnectedNN, self).__init__()
        # 5层结构：输入层→隐藏层1(7)→隐藏层2(6)→隐藏层3(5)→输出层(1)
        self.fc1 = nn.Linear(input_dim, 7)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(7, 6)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(6, 5)  # 隐藏层2到隐藏层3
        self.fc4 = nn.Linear(5, 1)  # 隐藏层3到输出层
        self.relu = nn.ReLU()  # 激活函数（避免梯度消失）

    def forward(self, x):
        # 前向传播流程
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # 回归任务输出层无激活
        return x


# ==== 4. 模型训练与可视化 ====
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device, scaler_y):
    # 初始化训练记录
    train_loss_history = []
    test_loss_history = []
    best_test_loss = float('inf')
    best_model_path = "best_footprint_model.pt"

    print(f"\n开始训练（设备: {device}），共{epochs}轮")
    for epoch in range(epochs):
        # ---------------------- 训练阶段 ----------------------
        model.train()
        train_total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 累加损失
            train_total_loss += loss.item() * inputs.size(0)

        # 计算平均训练损失
        train_avg_loss = train_total_loss / len(train_loader.dataset)
        train_loss_history.append(train_avg_loss)

        # ---------------------- 测试阶段 ----------------------
        model.eval()
        test_total_loss = 0.0
        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_total_loss += loss.item() * inputs.size(0)

        # 计算平均测试损失
        test_avg_loss = test_total_loss / len(test_loader.dataset)
        test_loss_history.append(test_avg_loss)

        # 保存最佳模型（测试损失最小）
        if test_avg_loss < best_test_loss:
            best_test_loss = test_avg_loss
            torch.save(model.state_dict(), best_model_path)
            # 每更新最佳模型时打印日志
            print(
                f"第{epoch + 1:3d}轮 | 训练损失: {train_avg_loss:.6f} | 测试损失: {test_avg_loss:.6f} | ✅ 更新最佳模型")
        else:
            # 每10轮打印一次普通日志
            if (epoch + 1) % 10 == 0:
                print(f"第{epoch + 1:3d}轮 | 训练损失: {train_avg_loss:.6f} | 测试损失: {test_avg_loss:.6f}")

    # ---------------------- 训练后可视化 ----------------------
    plt.figure(figsize=(14, 10))

    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss_history, label='训练损失', color='#1f77b4', linewidth=2)
    plt.plot(range(1, epochs + 1), test_loss_history, label='测试损失', color='#ff7f0e', linewidth=2)
    plt.xlabel('训练轮数', fontsize=11)
    plt.ylabel('MSE损失', fontsize=11)
    plt.title('训练与测试损失变化曲线', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. 训练集真实值vs预测值（反标准化）
    plt.subplot(2, 2, 2)
    model.eval()
    with torch.no_grad():
        # 训练集预测
        train_pred_norm = model(train_loader.dataset.X.to(device)).cpu().numpy()
        train_pred = scaler_y.inverse_transform(train_pred_norm)
        train_true = scaler_y.inverse_transform(train_loader.dataset.y.numpy())
    plt.scatter(train_true, train_pred, alpha=0.6, color='#2ca02c', s=50)
    plt.plot([train_true.min(), train_true.max()],
             [train_true.min(), train_true.max()],
             'r--', linewidth=2, label='理想拟合线')
    plt.xlabel('真实生态足迹', fontsize=11)
    plt.ylabel('预测生态足迹', fontsize=11)
    plt.title('训练集预测结果对比', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 3. 测试集真实值vs预测值（反标准化）
    plt.subplot(2, 2, 3)
    with torch.no_grad():
        # 测试集预测
        test_pred_norm = model(test_loader.dataset.X.to(device)).cpu().numpy()
        test_pred = scaler_y.inverse_transform(test_pred_norm)
        test_true = scaler_y.inverse_transform(test_loader.dataset.y.numpy())
    plt.scatter(test_true, test_pred, alpha=0.6, color='#d62728', s=50)
    plt.plot([test_true.min(), test_true.max()],
             [test_true.min(), test_true.max()],
             'r--', linewidth=2, label='理想拟合线')
    plt.xlabel('真实生态足迹', fontsize=11)
    plt.ylabel('预测生态足迹', fontsize=11)
    plt.title('测试集预测结果对比', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 4. 测试集误差分布
    plt.subplot(2, 2, 4)
    test_error = test_true - test_pred
    plt.hist(test_error, bins=25, color='#9467bd', alpha=0.7, edgecolor='black')
    plt.xlabel('预测误差（真实值-预测值）', fontsize=11)
    plt.ylabel('样本数量', fontsize=11)
    plt.title('测试集预测误差分布', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存可视化图像
    plt.tight_layout()
    plt.savefig("footprint_training_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n训练可视化图像已保存为: footprint_training_visualization.png")

    # 计算R²指标（评估拟合优度）
    def r2_score(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0

    train_r2 = r2_score(train_true, train_pred)
    test_r2 = r2_score(test_true, test_pred)

    # 输出训练结果
    print(f"\n==== 训练完成 ====")
    print(f"最佳模型保存路径: {best_model_path}")
    print(f"最佳测试损失: {best_test_loss:.6f}")
    print(f"训练集 - MSE: {train_avg_loss:.6f}, R²: {train_r2:.4f}")
    print(f"测试集 - MSE: {test_avg_loss:.6f}, R²: {test_r2:.4f}")

    return model, train_loss_history, test_loss_history, best_test_loss


# ==== 5. 模型加载与预测 ====
def load_model_and_predict(model_class, model_path, input_dim, scaler_X, scaler_y, new_sample):
    # 初始化模型结构
    model = model_class(input_dim=input_dim)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换为评估模式
    print(f"\n从 {model_path} 加载模型成功")

    # 新样本预处理与预测
    new_sample_norm = scaler_X.transform(new_sample)
    new_sample_tensor = torch.tensor(new_sample_norm, dtype=torch.float32)

    with torch.no_grad():
        pred_norm = model(new_sample_tensor).numpy()
        pred = scaler_y.inverse_transform(pred_norm)

    return pred


# ==== 6. 主函数（整合所有流程） ====
def main():
    # 配置参数（根据电脑配置调整）
    DATA_PATH = "countries.csv"  # 数据集路径（需确保文件在当前目录）
    BATCH_SIZE = 16  # 批量大小：16GB内存设32，8GB设16，4GB设8
    NUM_WORKERS = 0  # Windows系统设0（避免多进程报错），Linux/macOS设2-4
    EPOCHS = 200  # 训练轮数
    LEARNING_RATE = 0.001  # 学习率

    # 步骤1：数据预处理
    try:
        (X_train, y_train, X_test, y_test,
         scaler_X, scaler_y, feature_cols, target_col) = load_and_preprocess_data(DATA_PATH)
    except FileNotFoundError:
        print(f"错误：未找到数据集文件 {DATA_PATH}，请检查文件路径！")
        return
    except Exception as e:
        print(f"数据预处理出错: {str(e)}")
        return

    # 步骤2：构建Dataset和DataLoader
    train_dataset = CountryFootprintDataset(X_train, y_train)
    test_dataset = CountryFootprintDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True  # 加速GPU数据传输
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"\nDataLoader配置完成:")
    print(f"批量大小: {BATCH_SIZE}, 工作进程数: {NUM_WORKERS}")
    print(f"训练集批次数量: {len(train_loader)}, 测试集批次数量: {len(test_loader)}")

    # 步骤3：初始化模型、损失函数、优化器
    input_dim = len(feature_cols)  # 输入维度=特征数量
    model = FullyConnectedNN(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到GPU/CPU

    # 损失函数（回归任务用MSE）和优化器（Adam收敛更快）
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print(f"\n模型初始化完成:")
    print(f"输入维度: {input_dim}, 模型设备: {device}")
    print(f"损失函数: MSELoss, 优化器: Adam (lr={LEARNING_RATE})")

    # 步骤4：模型训练
    model, train_loss, test_loss, best_loss = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        scaler_y=scaler_y
    )

    # 步骤5：模型加载与预测示例（用测试集第一个样本）
    print(f"\n==== 预测示例 ====")
    # 取测试集第一个样本作为示例
    sample_idx = 0
    new_sample = X_test[sample_idx:sample_idx + 1]  # 原始特征（未标准化）
    true_value = scaler_y.inverse_transform(y_test[sample_idx:sample_idx + 1])[0][0]

    # 加载最佳模型并预测
    pred_value = load_model_and_predict(
        model_class=FullyConnectedNN,
        model_path="best_footprint_model.pt",
        input_dim=input_dim,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        new_sample=new_sample
    )[0][0]

    # 输出预测结果
    print(f"示例样本特征（前5个）: {new_sample[0][:5]}...")
    print(f"真实{target_col}: {true_value:.4f}")
    print(f"预测{target_col}: {pred_value:.4f}")
    print(f"预测误差: {abs(true_value - pred_value):.4f}")


# ==== 7. 启动程序（适配Windows多进程） ====
if __name__ == "__main__":
    # Windows系统多进程保护（避免DataLoader报错）
    import platform

    if platform.system() == "Windows":
        import multiprocessing

        multiprocessing.freeze_support()
    main()