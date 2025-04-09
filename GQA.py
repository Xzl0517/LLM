import torch
import torch.nn as nn
import math

class GQA(nn.Module):
    def __init__(self, d_model=512, head_dim=128, query_heads=8, kv_heads=4):
        super(GQA, self).__init__()
        self.g_num = None
        self.d_model = d_model
        self.head_dim = head_dim
        self.q_heads = query_heads
        self.kv_heads = kv_heads

        self.q = nn.Linear(d_model, head_dim * query_heads)
        self.k = nn.Linear(d_model, head_dim * kv_heads)
        self.v = nn.Linear(d_model, head_dim * kv_heads)

    def forward(self, x):
        b, x_num, _ = x.shape

        if self.q_heads % self.kv_heads != 0:
            raise ValueError(f"query_heads {self.q_heads} must be divisible by kv_heads {self.kv_heads}")
        self.g_num = self.q_heads // self.kv_heads
        x_q = self.q(x)  # 4 10 1024
        x_k = self.k(x)  # 4 10 512
        x_v = self.v(x)

        x_q = x_q.reshape(b, x_num, self.g_num, self.kv_heads, -1)  # 4 10 2 4 128 -> 4 2 4 10 128
        x_k = x_k.reshape(b, x_num, self.kv_heads, -1)  # 4 10 4 128 -> 4 4 10 128
        x_v = x_v.reshape(b, x_num, self.kv_heads, -1)

        x_q = x_q.transpose(1, 2).transpose(2, 3)  # 4 2 4 10 128
        x_k = x_k.transpose(1, 2).unsqueeze(1)  # 4 1 4 10 128
        x_v = x_v.transpose(1, 2).unsqueeze(1)

        x_qk = torch.matmul(x_q, x_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        score = torch.softmax(x_qk, dim=-1)  # 4 2 4 10 10
        out = torch.matmul(score, x_v)  # 4 2 4 10 128
        out = out.transpose(1, 3).reshape(b, x_num, -1)
        return out

bs, seq_len, d_model = 4, 10, 512
h = torch.randn(bs, seq_len, d_model)
gqa = GQA(d_model=d_model)
output = gqa(h)
print(output.shape)


