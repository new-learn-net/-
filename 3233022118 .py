# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------- 步骤1：用pandas读取train.csv数据 ----------------------
# 读取CSV文件（若文件不在当前工作目录，需替换为完整文件路径，如'C:/data/train.csv'）
df = pd.read_csv(r'C:\Users\34618\Desktop\open\train.csv')

# 数据预处理：删除缺失值（处理csv中"3530.15736917"对应的空y值）
df = df.dropna(subset=['x', 'y'])  # 仅保留x和y都非空的行
# 转换数据类型为float，确保后续计算正常
df['x'] = df['x'].astype(float)
df['y'] = df['y'].astype(float)

# 提取特征x和标签y（sklearn要求特征为二维数组，需用reshape(-1,1)转换）
X = df['x'].values.reshape(-1, 1)  # 特征矩阵（n行1列）
y_true = df['y'].values            # 真实标签（一维数组）

# ---------------------- 步骤2：训练y=wx+b线性模型 ----------------------
# 初始化线性回归模型（默认计算截距b，即fit_intercept=True）
model = LinearRegression()
# 拟合数据，得到最优参数w和b
model.fit(X, y_true)

# 提取训练得到的模型参数
w_opt = model.coef_[0]  # 斜率w（coef_返回数组，取第一个元素）
b_opt = model.intercept_  # 截距b

# 输出最优参数
print(f"训练得到的最优参数：")
print(f"斜率 w = {w_opt:.6f}")
print(f"截距 b = {b_opt:.6f}")
print(f"线性回归方程：y = {w_opt:.6f}x + {b_opt:.6f}")

# ---------------------- 步骤3：计算不同w、b对应的损失值（MSE） ----------------------
# 定义损失函数：均方误差（MSE），衡量预测值与真实值的偏差
def calculate_mse(w, b, X, y_true):
    y_pred = w * X.flatten() + b  # 计算预测值（X.flatten()转为一维数组便于计算）
    mse = mean_squared_error(y_true, y_pred)  # 计算MSE
    return mse

# 1. 生成w的取值范围（围绕最优w_opt，取±0.5的区间，生成50个点）
w_range = np.linspace(w_opt - 0.5, w_opt + 0.5, 50)
# 固定b为最优值b_opt，计算每个w对应的MSE
loss_w = [calculate_mse(w, b_opt, X, y_true) for w in w_range]

# 2. 生成b的取值范围（围绕最优b_opt，取±5的区间，生成50个点，b的敏感度低于w，区间稍大）
b_range = np.linspace(b_opt - 5, b_opt + 5, 50)
# 固定w为最优值w_opt，计算每个b对应的MSE
loss_b = [calculate_mse(w_opt, b, X, y_true) for b in b_range]

# ---------------------- 步骤4：用matplotlib绘制w-loss、b-loss关系图 ----------------------
# 设置绘图风格（使图表更美观）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False    # 支持负号显示
plt.figure(figsize=(12, 5))  # 设置画布大小（宽12英寸，高5英寸）

# 子图1：w与loss的关系
plt.subplot(1, 2, 1)  # 1行2列，第1个子图
plt.plot(w_range, loss_w, color='#1f77b4', linewidth=2)  # 绘制w-loss曲线
plt.scatter(w_opt, calculate_mse(w_opt, b_opt, X, y_true),
           color='red', s=50, label=f'最优w\nw={w_opt:.6f}\nloss={calculate_mse(w_opt, b_opt, X, y_true):.2f}')  # 标记最优w点
plt.xlabel('斜率 w', fontsize=12)
plt.ylabel('损失值 Loss (MSE)', fontsize=12)
plt.title('w与损失值（Loss）的关系', fontsize=14, fontweight='bold')
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格（透明度0.3）

# 子图2：b与loss的关系
plt.subplot(1, 2, 2)  # 1行2列，第2个子图
plt.plot(b_range, loss_b, color='#ff7f0e', linewidth=2)  # 绘制b-loss曲线
plt.scatter(b_opt, calculate_mse(w_opt, b_opt, X, y_true),
           color='red', s=50, label=f'最优b\nb={b_opt:.6f}\nloss={calculate_mse(w_opt, b_opt, X, y_true):.2f}')  # 标记最优b点
plt.xlabel('截距 b', fontsize=12)
plt.ylabel('损失值 Loss (MSE)', fontsize=12)
plt.title('b与损失值（Loss）的关系', fontsize=14, fontweight='bold')
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格

# 调整子图间距，避免标签重叠
plt.tight_layout()
# 显示图片（若需保存，可添加 plt.savefig('w_b_loss_relation.png', dpi=300, bbox_inches='tight')）
plt.show()