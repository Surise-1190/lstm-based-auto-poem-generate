"""
全局超参数配置
"""

# 训练参数
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001

# 模型参数
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2

# 数据参数
SEQ_LEN = 15  # 包含标点符号的诗句长度
DATA_PATH = 'data\poet.json'
MODEL_PATH = 'poem_lstm_model.pth'

# 设备
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")