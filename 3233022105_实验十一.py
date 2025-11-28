import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from collections import Counter
from tqdm import tqdm  # 进度条，方便查看大规模数据训练进度
import fasttext
import warnings

warnings.filterwarnings('ignore')


# ===================== 全局配置（可根据数据调整）=====================
class Config:
    # 数据路径
    TRAIN_PATH = "train.tsv"
    TEST_PATH = "test.tsv"
    # 模型参数
    MODEL_TYPE = "textrnn"  # 可选：textrnn / fasttext
    MAX_LEN = 64  # 句子最大长度（大规模数据建议64-128）
    VOCAB_MIN_FREQ = 2  # 词汇表最小词频
    EMBEDDING_DIM = 128  # 嵌入层维度
    HIDDEN_DIM = 128  # RNN隐藏层维度
    NUM_LAYERS = 2  # RNN层数
    BIDIRECTIONAL = True  # 是否双向RNN
    DROPOUT = 0.3  # dropout率
    NUM_CLASSES = 5  # 情感分类数（0-4）
    # 训练参数
    BATCH_SIZE = 128  # 批量大小（大规模数据建议128/256）
    LEARNING_RATE = 1e-3
    EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 保存路径
    SAVE_DIR = "model_output"
    os.makedirs(SAVE_DIR, exist_ok=True)


config = Config()


# ===================== 数据预处理（大规模数据优化）=====================
class LargeScaleTextDataset(Dataset):
    """
    适配大规模数据的Dataset（按需加载，避免一次性读入内存）
    """

    def __init__(self, file_path, vocab=None, is_train=True, max_len=config.MAX_LEN):
        self.file_path = file_path
        self.is_train = is_train
        self.max_len = max_len
        self.vocab = vocab
        self.data = self._load_data()
        # 训练集构建词汇表
        if is_train and vocab is None:
            self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab) if self.vocab else 0

    def _load_data(self):
        """分批加载TSV数据，避免内存溢出"""
        chunks = []
        for chunk in pd.read_csv(self.file_path, sep="\t", chunksize=10000):  # 按1万行分批读
            # 空值处理
            chunk = chunk.fillna({"Phrase": ""})
            if self.is_train:
                chunk = chunk[chunk["Sentiment"].isin(range(config.NUM_CLASSES))]  # 过滤无效标签
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def _build_vocab(self):
        """基于训练集构建词汇表（分批统计，适配大规模数据）"""
        vocab_counter = Counter()
        # 分批统计词频
        for idx in range(0, len(self.data), 10000):
            chunk_texts = self.data["Phrase"].iloc[idx:idx + 10000].values
            for text in chunk_texts:
                words = str(text).lower().split()
                vocab_counter.update(words)
        # 构建词汇表
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in vocab_counter.items():
            if freq >= config.VOCAB_MIN_FREQ:
                vocab[word] = len(vocab)
        return vocab

    def _text2seq(self, text):
        """文本转索引序列，含padding/truncation"""
        words = str(text).lower().split()
        seq = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        # 截断/填充
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq += [self.vocab["<PAD>"]] * (self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["Phrase"]
        seq = self._text2seq(text)
        if self.is_train:
            label = torch.tensor(self.data.iloc[idx]["Sentiment"], dtype=torch.long)
            return seq, label
        else:
            phrase_id = torch.tensor(self.data.iloc[idx]["PhraseId"], dtype=torch.long)
            return seq, phrase_id


# 加载数据
print("加载训练集...")
train_dataset = LargeScaleTextDataset(config.TRAIN_PATH, is_train=True)
vocab = train_dataset.vocab  # 共享词汇表
print("加载测试集...")
test_dataset = LargeScaleTextDataset(config.TEST_PATH, vocab=vocab, is_train=False)

# 数据加载器参数设置（修复prefetch_factor冲突 + 避免batch_size重复）
num_workers = 4 if sys.platform != "win32" else 0
# 基础加载器参数（包含batch_size）
base_loader_kwargs = {
    "batch_size": config.BATCH_SIZE,
    "num_workers": num_workers,
    "pin_memory": True if config.DEVICE.type == "cuda" else False  # 只有GPU时启用pin_memory
}
# 仅在多进程模式下添加prefetch_factor
if num_workers > 0:
    base_loader_kwargs["prefetch_factor"] = 2

# 训练集加载器（添加shuffle=True）
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **base_loader_kwargs  # 这里包含batch_size，不再重复传
)
# 测试集加载器（shuffle=False）
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    **base_loader_kwargs
)


# ===================== 模型定义（TextRNN）=====================
class TextRNN(nn.Module):
    def __init__(self, vocab_size):
        super(TextRNN, self).__init__()
        # 嵌入层（padding忽略）
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=vocab["<PAD>"]
        )
        # 双向GRU/LSTM（可选）
        self.rnn = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            bidirectional=config.BIDIRECTIONAL,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )
        # 全连接层（双向则维度*2）
        rnn_output_dim = config.HIDDEN_DIM * 2 if config.BIDIRECTIONAL else config.HIDDEN_DIM
        self.fc = nn.Linear(rnn_output_dim, config.NUM_CLASSES)
        # Dropout层
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        # x: [batch_size, max_len]
        embed = self.embedding(x)  # [batch_size, max_len, embed_dim]
        embed = self.dropout(embed)
        # RNN前向传播
        rnn_out, _ = self.rnn(embed)  # [batch_size, max_len, rnn_output_dim]
        # 取最后一个时间步的输出（也可取均值/最大值）
        final_out = rnn_out[:, -1, :]  # [batch_size, rnn_output_dim]
        # 分类
        logits = self.fc(final_out)  # [batch_size, num_classes]
        return logits


# ===================== 训练/验证/预测工具函数=====================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch + 1}/{config.EPOCHS}")
    for seq, label in pbar:
        seq, label = seq.to(config.DEVICE), label.to(config.DEVICE)
        # 前向传播
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits, label)
        # 反向传播
        loss.backward()
        optimizer.step()
        # 统计
        total_loss += loss.item() * seq.size(0)
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == label).sum().item()
        total_samples += seq.size(0)
        # 更新进度条
        pbar.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "acc": f"{total_correct / total_samples:.4f}"
        })
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc="Validate")
    for seq, label in pbar:
        seq, label = seq.to(config.DEVICE), label.to(config.DEVICE)
        logits = model(seq)
        loss = criterion(logits, label)
        # 统计
        total_loss += loss.item() * seq.size(0)
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == label).sum().item()
        total_samples += seq.size(0)
        pbar.set_postfix({
            "val_loss": f"{total_loss / total_samples:.4f}",
            "val_acc": f"{total_correct / total_samples:.4f}"
        })
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def predict(model, loader):
    model.eval()
    predictions = []
    phrase_ids = []
    pbar = tqdm(loader, desc="Predict")
    for seq, phrase_id in pbar:
        seq = seq.to(config.DEVICE)
        logits = model(seq)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(pred)
        phrase_ids.extend(phrase_id.numpy())
    return phrase_ids, predictions


# ===================== FastText 实现（大规模数据适配）=====================
def train_fasttext():
    """FastText训练（分批生成训练文件，避免内存溢出）"""
    # 生成FastText格式的训练文件
    fasttext_train_path = os.path.join(config.SAVE_DIR, "fasttext_train.txt")
    # 分批写入
    with open(fasttext_train_path, "w", encoding="utf-8") as f:
        for idx in range(0, len(train_dataset.data), 10000):
            chunk = train_dataset.data.iloc[idx:idx + 10000]
            for _, row in chunk.iterrows():
                label = f"__label__{row['Sentiment']}"
                text = str(row["Phrase"]).lower().replace("\n", " ")
                f.write(f"{label} {text}\n")

    # 训练FastText模型
    print("开始训练FastText模型...")
    model = fasttext.train_supervised(
        input=fasttext_train_path,
        lr=config.LEARNING_RATE,
        dim=config.EMBEDDING_DIM,
        ws=5,
        epoch=config.EPOCHS,
        minCount=config.VOCAB_MIN_FREQ,
        wordNgrams=2,
        loss="softmax",
        bucket=200000,
        thread=os.cpu_count(),  # 用满CPU核心
        verbose=2
    )

    # 测试集预测
    fasttext_test_path = os.path.join(config.SAVE_DIR, "fasttext_test.txt")
    with open(fasttext_test_path, "w", encoding="utf-8") as f:
        for idx in range(0, len(test_dataset.data), 10000):
            chunk = test_dataset.data.iloc[idx:idx + 10000]
            for _, row in chunk.iterrows():
                text = str(row["Phrase"]).lower().replace("\n", " ")
                f.write(f"{text}\n")

    # 预测
    predictions = []
    phrase_ids = test_dataset.data["PhraseId"].values
    with open(fasttext_test_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="FastText Predict"):
            pred = model.predict(line.strip())[0][0]
            predictions.append(int(pred.replace("__label__", "")))

    # 保存结果
    submission = pd.DataFrame({
        "PhraseId": phrase_ids,
        "Sentiment": predictions
    })
    submission_path = os.path.join(config.SAVE_DIR, "fasttext_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"FastText预测结果已保存至: {submission_path}")
    # 保存模型
    model.save_model(os.path.join(config.SAVE_DIR, "fasttext_model.bin"))
    return model


# ===================== 主训练流程=====================
def main():
    if config.MODEL_TYPE == "fasttext":
        # FastText训练
        train_fasttext()
    elif config.MODEL_TYPE == "textrnn":
        # TextRNN训练
        print(f"使用设备: {config.DEVICE}")
        model = TextRNN(vocab_size=train_dataset.vocab_size).to(config.DEVICE)
        # 损失函数+优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率衰减

        # 训练循环
        best_acc = 0.0
        for epoch in range(config.EPOCHS):
            # 训练
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
            # 验证（用训练集的10%做验证，大规模数据可单独拆分验证集）
            val_size = int(len(train_loader.dataset) * 0.1)
            val_dataset = torch.utils.data.Subset(train_loader.dataset, range(val_size))
            # 验证集加载器（使用基础参数，shuffle=False）
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **base_loader_kwargs
            )
            val_loss, val_acc = validate(model, val_loader, criterion)
            # 学习率衰减
            scheduler.step()

            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "vocab": vocab
                }, os.path.join(config.SAVE_DIR, "best_textrnn_model.pth"))
                print(f"保存最优模型，验证准确率: {best_acc:.4f}")

        # 加载最优模型预测（修复设备不匹配问题）
        checkpoint = torch.load(
            os.path.join(config.SAVE_DIR, "best_textrnn_model.pth"),
            map_location=config.DEVICE
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        phrase_ids, predictions = predict(model, test_loader)

        # 保存预测结果
        submission = pd.DataFrame({
            "PhraseId": phrase_ids,
            "Sentiment": predictions
        })
        submission_path = os.path.join(config.SAVE_DIR, "textrnn_submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"TextRNN预测结果已保存至: {submission_path}")
    else:
        raise ValueError("MODEL_TYPE 仅支持 textrnn / fasttext")


if __name__ == "__main__":
    main()