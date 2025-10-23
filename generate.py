import torch
import numpy as np

from config import device, SEQ_LEN

def generate_poem(model, word_to_idx, idx_to_word, start_words="", max_len=100):
    """
    生成一首古诗
    :param model: 训练好的模型
    :param word_to_idx: 字到索引的映射
    :param idx_to_word: 索引到字的映射
    :param start_words: 起始字（如 "春"）
    :param max_len: 最大生成长度，防止无限生成
    :return: 生成的诗歌字符串（不含 \n，遇到句号自动停止）
    """
    model.eval()
    vocab_size = len(idx_to_word)
    generated = ""

    # === 输入初始化 ===
    if start_words:
        input_seq = [word_to_idx.get(ch, 0) for ch in start_words]
        generated = start_words
    else:
        idx = np.random.randint(vocab_size)
        input_seq = [idx]
        generated = idx_to_word[idx]

    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
    hidden = None

    # === 生成循环 ===
    with torch.no_grad():
        for _ in range(max_len - len(generated)):
            output, hidden = model(input_tensor, hidden)
            pred = output[0, -1, :]  # 最后一个时间步的输出
            pred = torch.softmax(pred, dim=0)  # 转为概率
            pred_idx = torch.multinomial(pred, 1).item()  # 采样
            char = idx_to_word.get(pred_idx, '')

            # === 关键：终止条件 ===
            if char in "。！？":
                generated += char
                break  # 遇到句号类标点，立即停止生成

            # === 忽略无效字符（如 \n，如果不在 vocab 中）===
            if char in "\n\r\t":  # 如果模型意外生成了这些字符
                continue  # 跳过，不加入输出

            generated += char
            # 更新输入：只输入最新一个字
            input_tensor = torch.tensor([[pred_idx]], dtype=torch.long).to(device)

    # === 格式化输出：按句分行 ===
    lines = []
    current_line = ""
    for ch in generated:
        if ch in "。！？":
            current_line += ch
            lines.append(current_line.strip())
            current_line = ""
        elif ch in "，；：":
            current_line += ch
        else:
            current_line += ch

    # 处理最后一句（可能没有标点）
    if current_line.strip():
        lines.append(current_line.strip())

    # 用换行符连接（这是输出时加的，不是模型生成的）
    return "\n".join(lines)