import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data = pd.read_csv('train.csv')
x = torch.tensor(data[['x']].values, dtype=torch.float32)
y = torch.tensor(data[['y']].values, dtype=torch.float32)


# 定义模型
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.normal_(self.linear.weight, 0.0, 0.02)
        nn.init.normal_(self.linear.bias, 0.0, 0.02)

    def forward(self, x): return self.linear(x)


# 训练函数
def train(opt_name, lr=0.01, epochs=200):
    model = LinearModel()
    opt = getattr(optim, opt_name)(model.parameters(), lr)
    loss_fn = nn.MSELoss()

    losses, w_hist, b_hist = [], [], []
    for _ in range(epochs):
        loss = loss_fn(model(x), y)
        opt.zero_grad();
        loss.backward();
        opt.step()
        losses.append(loss.item())
        w_hist.append(model.linear.weight.item())
        b_hist.append(model.linear.bias.item())
    return losses, w_hist, b_hist


# 三种优化器性能对比
opts = ['Adam', 'ASGD', 'RMSprop']
results = {opt: train(opt) for opt in opts}

plt.figure(figsize=(10, 5))
for opt in opts: plt.plot(results[opt][0], label=opt)
plt.title('不同优化器损失下降曲线');
plt.xlabel('Epoch');
plt.ylabel('Loss')
plt.legend();
plt.grid();
plt.show()

# w/b 参数变化可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for opt in opts:
    ax[0].plot(results[opt][1], label=opt)
    ax[1].plot(results[opt][2], label=opt)
ax[0].set(title='w 参数变化', xlabel='Epoch', ylabel='w')
ax[1].set(title='b 参数变化', xlabel='Epoch', ylabel='b')
[a.legend() and a.grid() for a in ax]
plt.show()

# 调节 Epoch 和学习率
for title, params, param_name in [
    ('不同 epoch 对损失的影响', [50, 100, 300], 'epochs'),
    ('不同学习率 对损失的影响', [0.001, 0.01, 0.1], 'lr')
]:
    plt.figure(figsize=(10, 5))
    for p in params:
        kwargs = {param_name: p} if param_name == 'epochs' else {'lr': p}
        losses, _, _ = train('Adam', **kwargs)
        plt.plot(losses, label=f'{param_name[:-1]}={p}')
    plt.title(title);
    plt.xlabel('Epoch');
    plt.ylabel('Loss')
    plt.legend();
    plt.grid();
    plt.show()