import torch
import torch.nn as nn
import math


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


class MLA(nn.Module):
    def __init__(self, d_model=256, down_dim=64, up_dim=128, num_heads=8, rope_head_dim=26, dropout_prob=0.0):
        super(MLA, self).__init__()

        self.d_model = d_model
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = up_dim // num_heads

        self.down_proj_kv = nn.Linear(d_model, down_dim)
        self.down_proj_q = nn.Linear(d_model, down_dim)

        self.up_proj_k = nn.Linear(down_dim, up_dim)
        self.up_proj_v = nn.Linear(down_dim, up_dim)
        self.up_proj_q = nn.Linear(down_dim, up_dim)

        self.proj_qr = nn.Linear(down_dim, rope_head_dim * num_heads)
        self.proj_kr = nn.Linear(d_model, rope_head_dim)

        self.rope_q = RotaryEmbedding(rope_head_dim * num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_head_dim, 1)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_heads * self.v_head_dim, d_model)
        self.res_dropout = nn.Dropout(dropout_prob)

    def forward(self, h, mask=None):

        bs, seq_len, _ = h.size()

        c_t_kv = self.down_proj_kv(h)  # [bs, seq_len, down_dim]
        k_t_c = self.up_proj_k(c_t_kv)  # [bs, seq_len, up_dim]
        v_t_c = self.up_proj_v(c_t_kv)  # [bs, seq_len, up_dim]
        c_t_q = self.down_proj_q(h)  # [bs, seq_len, down_dim]
        q_t_c = self.up_proj_q(c_t_q)  # [bs, seq_len, up_dim]
        # 对q应用RoPE
        q_t_r = self.proj_qr(c_t_q)  # [bs, seq_len, rope_head_dim*num_heads]
        q_t_r = q_t_r.view(bs, seq_len, self.num_heads, self.rope_head_dim).transpose(1, 2)  #  [bs, num_heads, seq_len, rope_head_dim]
        q_t_r = self.rope_q(q_t_r)
        # 对k应用RoPE
        k_t_r = self.proj_kr(h)  # [bs, seq_len, rope_head_dim]
        k_t_r = k_t_r.unsqueeze(1)  # [bs, 1, seq_len, rope_head_dim]
        k_t_r = self.rope_k(k_t_r)

        q_t_c = q_t_c.view(bs, seq_len, self.num_heads, -1).transpose(1,2)  # [bs, num_heads, seq_len, up_dim/num_heads]
        q = torch.cat([q_t_c, q_t_r], dim=-1)  # [bs, num_heads, seq_len, (up_dim+rope_head_dim)/num_heads]

        k_t_c = k_t_c.view(bs, seq_len, self.num_heads, -1).transpose(1,2)  # [bs, num_heads, seq_len, up_dim/num_heads]
        k_t_r = k_t_r.expand(bs, self.num_heads, seq_len, -1)  # [bs, num_heads, seq_len, rope_head_dim]
        k = torch.cat([k_t_c, k_t_r], dim=-1)  # [bs, num_heads, seq_len, (up_dim+rope_head_dim)/num_heads]

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-1, -2))  # [bs, num_heads, seq_len, seq_len]
        scores = scores / (math.sqrt(self.head_dim) + math.sqrt(self.rope_head_dim))

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))  # [bs, num_heads, seq_len, seq_len]

        attn_weights = torch.softmax(scores, dim=-1)  # [bs, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        # V维度调整
        v_t_c = v_t_c.view(bs, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)  # [bs, num_heads, seq_len, v_head_dim]

        context = torch.matmul(attn_weights, v_t_c)  # [bs, num_heads, seq_len, v_head_dim]
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, -1)  # [bs, seq_len, num_heads*v_head_dim]

        # 输出投影
        output = self.fc(context)  # [bs, seq_len, d_model]
        output = self.res_dropout(output)

        return output


if __name__ == '__main__':

    bs, seq_len, d_model = 4, 10, 512
    h = torch.randn(bs, seq_len, d_model)
    mla = MLA(d_model=d_model)
    output = mla(h)
    print(output.shape)
