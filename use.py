import os
import torch

from config import *
from data.dataset import PoemDataset
from model.lstm_model import PoetryLSTM
from generate import *

def load_model_and_vocab(model_path, data_path):
    """
    加载训练好的模型和词汇表
    """
    # 1. 先创建 Dataset 以获取词汇表（不需要完整数据，只需 build_vocab）
    dataset = PoemDataset(data_path)
    vocab_size = len(dataset.words)
    word_to_idx = dataset.word_to_idx
    idx_to_word = dataset.idx_to_word

    # 2. 创建模型结构
    model = PoetryLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBEDDING_DIM,      # 注意：这些参数必须和训练时一致！
        hidden_dim=HIDDEN_DIM,
        num_layers=2
    ).to(device)

    # 3. 加载模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 模型已从 {model_path} 加载")
        model.eval()  # 切换到评估模式
        return model, word_to_idx, idx_to_word
    else:
        raise FileNotFoundError(f"❌ 模型文件未找到: {model_path}")

def main():
    print(f"使用设备: {device}")

    try:
        # 直接加载模型（不训练）
        model, word_to_idx, idx_to_word = load_model_and_vocab(MODEL_PATH, DATA_PATH)

        print("\n=== 开始生成诗歌 ===")
        for i in range(5):
            # poem = generate(model, word_to_idx, idx_to_word, start_words="花",char_len=5,poem_max_len=24)
            poem = generate_poem(model, word_to_idx, idx_to_word, start_words="花")
            print(poem)
            print("-" * 20)

    except Exception as e:
        print(f"❌ 加载失败: {e}")

if __name__ == "__main__":
    main()