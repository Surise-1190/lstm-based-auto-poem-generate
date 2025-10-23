import os
from config import *
from train_with_scheduler import train
from generate import generate_poem
from data.dataset import PoemDataset
from model.lstm_model import PoetryLSTM
import torch

def main():
    print(f"使用设备: {device}")

    model, word_to_idx, idx_to_word = train()

    # 生成诗歌
    print("\n=== 自动生成诗歌 ===")
    for i in range(5):
        poem = generate_poem(model, word_to_idx, idx_to_word, start_words="春")
        print(poem)
        print("-" * 20)

if __name__ == "__main__":
    main()