import torch
import torch.nn as nn

class PoetryLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(PoetryLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.5 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden