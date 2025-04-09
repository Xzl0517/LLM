import torch
import torch.nn as nn
import math


class MQA(nn.Module):
    def __init__(self, d_model=512, head_dim=128, num_heads=8, dropout_prob=0.1):
        super(MQA, self).__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.q = nn.Linear(d_model, num_heads * head_dim)
        self.k = nn.Linear(d_model, head_dim)
        self.v = nn.Linear(d_model, head_dim)

    def forward(self, x):
        b, x_num, _ = x.shape
        x_q = self.q(x)
        x_k = self.k(x)  # 4 10 128
        x_v = self.v(x)  # 4 10 128

        x_q = x_q.reshape(b, x_num, self.num_heads, -1).transpose(1, 2)  # 4 8 10 128
        x_qk = torch.matmul(x_q, x_k.transpose(1, 2).unsqueeze(1)) / math.sqrt(self.head_dim)
        score = torch.softmax(x_qk, dim=-1)
        out = torch.matmul(score, x_v.unsqueeze(1))
        out = out.transpose(1, 2).reshape(b, x_num, -1)
        return out


bs, seq_len, d_model = 4, 10, 512
h = torch.randn(bs, seq_len, d_model)
mqa = MQA(d_model=d_model)
output = mqa(h)
print(output)
