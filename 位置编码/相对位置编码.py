import torch
import torch.nn as nn


class RelativePosition(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.num_units = d_model
        self.max_relative_position = seq_len
        self.embeddings_table = nn.Parameter(torch.Tensor(seq_len * 2 + 1, d_model))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


seq_len, d_model = 32, 60
q = torch.randn((seq_len, d_model))
k = torch.randn((seq_len, d_model))
pos_encoding = RelativePosition(seq_len=seq_len, d_model=d_model)

PE = pos_encoding(q.shape[0], k.shape[0])
