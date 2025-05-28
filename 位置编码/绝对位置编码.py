import torch
from torch import nn


class PositionalEncoding(nn.Module):
    # num+hiddens:向量长度  max_len:序列最大长度
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        seq_len, d_model = x.shape
        pos_num = torch.arange(0, seq_len).reshape(-1, 1)  # 为每个token生成序列号 seq_len x 1
        P = 10000 ** torch.arange(0, d_model, 1) / d_model  # d_model 大小

        PE = pos_num / P
        PE[:, 0::2] = PE[:, 0::2].sin()
        PE[:, 1::2] = PE[:, 1::2].cos()
        print(PE.shape)
        return x + PE


seq_len, d_model = 32, 60
x = torch.randn((seq_len, d_model))
pos_encoding = PositionalEncoding()

X = pos_encoding(x)
