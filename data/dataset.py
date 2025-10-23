import torch
import os
import json
import numpy as np
from collections import Counter
from torch.utils.data import Dataset


class PoemDataset(Dataset):
    def __init__(self, data_path, seq_len=15):
        """
        :param data_path: JSON 文件路径
        :param seq_len: 每个训练序列的长度（如 15 字）
        """
        self.seq_len = seq_len
        self.data_path = data_path

        # 提取所有诗句
        self.sentences = self.load_sentences_from_json()

        # 构建词汇表
        self.words, self.word_to_idx, self.idx_to_word = self.build_vocab()

        # 将所有句子转为索引序列，并截断/补全到 seq_len+1（用于生成 x 和 y）
        self.data = self.prepare_data()

    def load_sentences_from_json(self):
        """从 JSON 文件中提取所有诗句，并清洗特殊字符"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件未找到: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            poems = json.load(f)

        sentences = []
        # 定义要过滤的控制字符
        control_chars = {'\n', '\r', '\t', '\u3000'}  # \u3000 是中文全角空格

        for poem in poems:
            for line in poem.get("paragraphs", []):
                # 1. 移除所有控制字符
                cleaned_line = ''.join(ch for ch in line if ch not in control_chars)
                # 2. 去除首尾空白
                cleaned_line = cleaned_line.strip()
                # 3. 过滤空行
                if cleaned_line:
                    sentences.append(cleaned_line)
        return sentences

    def build_vocab(self):
        """基于所有诗句构建词汇表"""
        # 将所有句子拼接成一个大字符串
        all_text = "".join(self.sentences)

        # 统计字符频次
        counter = Counter(all_text)
        vocab = sorted(counter.keys())  # 按 Unicode 排序

        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

        return list(vocab), word_to_idx, idx_to_word

    def prepare_data(self):
        """将句子转换为固定长度的索引序列，用于训练"""
        data = []
        for sentence in self.sentences:
            # 截断或补全到 seq_len + 1 字（因为我们要用前 n 预测后 n）
            seq = sentence[:self.seq_len + 1]  # 最多取 seq_len+1 个字
            if len(seq) < self.seq_len + 1:
                # 补全（可用空格或其他填充符，也可不用）
                seq = seq + ' ' * (self.seq_len + 1 - len(seq))

            # 转为索引
            idx_seq = [self.word_to_idx.get(ch, 0) for ch in seq]
            data.append(idx_seq)

        return np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回一个训练样本：输入 x 和目标 y
        x: 前 n 个字（长度为 seq_len）
        y: 后 n 个字（预测下一个字）
        """
        seq = self.data[idx]  # 长度为 seq_len + 1
        x = seq[:-1]  # 输入：前 seq_len 个字
        y = seq[1:]  # 输出：从第2个字开始，预测下一个字
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    


if __name__ == "__main__":
    dataset = PoemDataset("D:\python_codes\基于LSTM的自动写诗\data\dataset.py", seq_len=15)
    print("词汇表大小:", len(dataset.words))
    print("前3个句子:", dataset.sentences[:3])
    x, y = dataset[0]
    print("x:", x.tolist())
    print("y:", y.tolist())