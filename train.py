from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from data.dataset import PoemDataset
from model.lstm_model import PoetryLSTM

def train():
    print("开始训练新模型...")

    # 构建数据集和加载器
    dataset = PoemDataset(DATA_PATH, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    vocab_size = len(dataset.words)
    print(f"词汇表大小: {vocab_size}")

    # 构建模型
    model = PoetryLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至: {MODEL_PATH}")

    return model, dataset.word_to_idx, dataset.idx_to_word