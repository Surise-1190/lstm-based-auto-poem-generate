from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

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

    # === 1. 数据集划分：训练集 + 验证集 ===
    dataset = PoemDataset(DATA_PATH, SEQ_LEN)
    
    # 划分 90% 训练，10% 验证
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    vocab_size = len(dataset.words)
    print(f"词汇表大小: {vocab_size}")
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    # === 2. 构建模型 ===
    model = PoetryLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # === 3. 早停机制 ===
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # 最多容忍 15 个 epoch 没提升

    # === 4. 训练循环 ===
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # --- 训练阶段 ---
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                output, _ = model(x_val)
                loss = criterion(output.reshape(-1, vocab_size), y_val.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- 打印日志 ---
        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

        # --- 学习率调度 ---
        scheduler.step(avg_val_loss)

        # --- 早停判断 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 只保存最佳模型
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ 模型已保存 (Val Loss 改进: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ 验证损失未改进，耐心计数: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"🎯 早停触发！最佳验证损失: {best_val_loss:.4f}")
            break

    print(f"训练完成，最终最佳模型已保存至: {MODEL_PATH}")
    return model, dataset.word_to_idx, dataset.idx_to_word