""" src/model.py """
import re
import torch
import torch.nn as nn
import json

from pathlib import Path
from config import DEVICE
from src.utils import debug, save_model_name
from src.attention import Attention

class Nya_Nya_ChineseEmotionLSTM(nn.Module):
    """ LSTM模型类 """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        debug.model('正在初始化LSTM模型')
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding (
            num_embeddings=vocab_size, # 要嵌入的单词总数
            embedding_dim=embedding_dim, # 词向量维度
            padding_idx=0 # 数字 0 作为填充符
        )
        # RNN LSTM层
        self.lstm = nn.LSTM (
            input_size=embedding_dim, # 输入大小（维度）
            hidden_size=hidden_dim, # 隐层大小（维度）
            num_layers=num_layers, # LSTM层数
            batch_first=True, # 输入形状为 (batch, seq_len, feature) 否则为(seq_len, batch, feature)
            bidirectional=True, # 开启双向LSTM
            dropout=0.5 if num_layers > 1 else 0 # 正则化率
        )
        # 线性分类层 双层LSTM会导致两倍的输出维度
        self.linear = nn.Linear(hidden_dim * 2, 7) # 七分类
        # 注意力层
        self.attention = Attention(hidden_dim * 2)

        # 正则化
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """前向传播"""
        x = self.embedding(x)
        x = self.dropout(x)
        x, (h, c) = self.lstm(x)
        x, w = self.attention(x)
        x = self.linear(x)
        return x

def load_model(model_path, sava_path=True, vocab_size=None, embedding_dim=None, hidden_dim=None, num_layers=None, find_best=False):
    """ 加载模型 """
    if find_best:
        max_accuracy = 0
        best_model_path = None
        
        # 寻找最佳准确率的模型
        for model_file in Path(model_path).glob('*.pth'):
            try:
                checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=True)
                accuracy = checkpoint.get("accuracy", 0)
            except: # 损坏的模型直接跳过
                continue
            # 对比找到准确率最大的模型
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_model_path = model_file
        
        if best_model_path is None:
            raise ValueError(f"在 {model_path} 中没有找到有效的模型文件")

        checkpoint = torch.load(best_model_path, map_location=DEVICE)
    else:
        checkpoint = torch.load(model_path, map_location=DEVICE)

    actual_vocab_size = checkpoint.get('vocab_size', vocab_size)
    actual_embedding_dim = checkpoint.get('embedding_dim', embedding_dim)
    actual_hidden_dim = checkpoint.get('hidden_dim', hidden_dim)
    actual_num_layers = checkpoint.get('num_layers', num_layers)
    
    model = Nya_Nya_ChineseEmotionLSTM(vocab_size=actual_vocab_size,
        embedding_dim=actual_embedding_dim,
        hidden_dim=actual_hidden_dim,num_layers=actual_num_layers
    )
    model.to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    if find_best:
        # 保存模型名
        if sava_path:
            save_model_name('session_state.json', best_model_path.name)

        print(f"已自动加载最佳模型: {best_model_path.name}")
        print(f"模型准确率: {max_accuracy * 100:.2f}%")
    else:
        if sava_path:
            save_model_name('session_state.json', model_path.name)

        print(f"已加载模型: {model_path.name}")
        print(f"模型准确率: {checkpoint['accuracy'] * 100:.2f}%")
    
    print(f"模型参数: vocab={actual_vocab_size}, embed={actual_embedding_dim}, "
          f"hidden={actual_hidden_dim}, layers={actual_num_layers}")
    print('-' * 40)
    
    return model