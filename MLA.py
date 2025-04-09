import torch
import torch.nn as nn
import math


class MLA(nn.Module):
    def __init__(self, d_model=512, down_dim=128, up_dim=256, num_heads=8, rope_head_dim=26, dropout_prob=0.1):
        super(MLA, self).__init__()

        self.d_model = d_model
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = up_dim // num_heads
        # 初始化kv联合以及q对应的dow,up projection
        self.down_proj_kv = nn.Linear(d_model, down_dim)  # W^{DKV}
        self.up_proj_k = nn.Linear(down_dim, up_dim)  # W^{UK}
        self.up_proj_v = nn.Linear(down_dim, up_dim)  # W^{UV}
        self.down_proj_q = nn.Linear(d_model, down_dim)  # W^{DQ}
        self.up_proj_q = nn.Linear(down_dim, up_dim)  # W^{UQ}
        # 初始化解耦的q,k进行MQA计算的映射矩阵
        self.proj_qr = nn.Linear(down_dim, rope_head_dim * num_heads)
        self.proj_kr = nn.Linear(d_model, rope_head_dim * 1)
        # 初始化解耦的q,k对应的rope类，因为头的数量不同，初始化2个实例
        self.rope_q = RotaryEmbedding(rope_head_dim * num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_head_dim, 1)
        # Dropout and final linear layer
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_heads * self.v_head_dim, d_model)
        self.res_dropout = nn.Dropout(dropout_prob)

    def forward(self, h, mask=None):
        bs, seq_len, _ = h.size()
        # setp1 :低秩转换
        c_t_kv = self.down_proj_kv(h)
        k_t_c = self.up_proj_k(c_t_kv)
        v_t_c = self.up_proj_v(c_t_kv)
        c_t_q = self.down_proj_q(h)
        q_t_c = self.up_proj_q(c_t_q)

        # step2:解耦的q,k进行MQA计算，同时引入ROPE
        # q_t_r,k_t_r施加rope时均扩展了n_h_r维度->[bs,n_h_r,seq_len,rope_head_dim]
        q_t_r = self.rope_q(self.proj_qr(c_t_q))
        k_t_r = self.rope_k(self.proj_kr(h))

        # step3:拼接step1，step2得到的q,k,进行sdpa计算
        # q_t_c扩展出num_heads为4维，以便于和q_t_r拼接
        q_t_c = q_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        # head_dim,rope_head_dim拼接
        q = torch.cat([q_t_c, q_t_r], dim=-1)
        # k_t_c扩展出num_heads为4维，以便于和k_t_r拼接
        k_t_c = k_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        # k_t_r为MQA,n_h_k_r=1,为了和q_t_r计算，需要在n_h_k_r维度复制
        # k_t_r:[bs,n_h_r_k,seq_len,rope_head_dim]->[bs,num_heads,seq_len,rope_head_dim]
        k_t_r = k_t_r.repeat(1, self.num_heads, 1, 1)
        # head_dim,rope_head_dim拼接
        k = torch.cat([k_t_c, k_t_r], dim=-1)
        # 注意力计算,[bs,num_heads,seq_len,seq_len]
        scores = torch.matmul(q, k.transpose(-1, -2))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores / (math.sqrt(self.head_dim) + math.sqrt(self.rope_head_dim)), dim=-1)
        scores = self.dropout(scores)
        # v_t_c和scores计算，扩展出num_heads维度
        v_t_c = v_t_c.reshape(bs, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        output = torch.matmul(scores, v_t_c)
        # 压缩num_head,送入最终统一映射层
        output = output.transpose(1, 2).reshape(bs, seq_len, -1)
        output = self.fc(output)
        output = self.res_dropout(output)
        return output


bs, seq_len, d_model = 4, 10, 512
h = torch.randn(bs, seq_len, d_model)
mla = MLA(d_model=d_model)
output = mla(h)

