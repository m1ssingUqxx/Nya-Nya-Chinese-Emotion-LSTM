""" src/attention """
import torch
import torch.nn.functional as F

from src.utils import debug

class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        debug.model('正在初始化注意力机制')
        super().__init__()
        # 计算注意力得分
        self.attention = torch.nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = self.attention(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)
        return context.squeeze(1), attn_weights