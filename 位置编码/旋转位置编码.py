import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_size, num_heads, base=10000, max_len=512):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.base = base
        self.max_len = max_len
        self.cos_pos, self.sin_pos = self.compute_pos_emb()

    def compute_pos_emb(self):
        theta_i = 1. / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(self.max_len)
        pos_emb = positions.unsqueeze(1) * theta_i.unsqueeze(0)

        cos_pos = pos_emb.sin().repeat_interleave(2, dim=-1)
        sin_pos = pos_emb.cos().repeat_interleave(2, dim=-1)
        return cos_pos, sin_pos

    def forward(self, q):
        bs, _, seq_len, _ = q.shape
        cos_pos = self.cos_pos[:seq_len].to(q.device)  # [seq_len, head_dim]
        sin_pos = self.sin_pos[:seq_len].to(q.device)  # [seq_len, head_dim]

        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

        # RoPE变换
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)  # 奇偶交替
        q2 = q2.reshape(q.shape).contiguous()

        return q * cos_pos + q2 * sin_pos


bs, num_head, seq_len, head_dim = 4, 8, 10, 64
h = torch.randn(bs, num_head, seq_len, head_dim)
Rope = RotaryEmbedding(head_dim*num_head, num_head)
output = Rope(h)
print(output.shape)