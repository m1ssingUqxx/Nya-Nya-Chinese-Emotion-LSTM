""" src/dataset.py """
import jieba
import pandas as pd
import torch
import os

from torch.utils.data import Dataset
from config import EMOTION_LABELS, STOP_SYNCS, TRAIN_DATA_DIR
from src.utils import debug

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

class SentimentDataSet(Dataset):
    def __init__(self, filedir, tokenizer=None, num_words=5000, max_len=100):
        debug.model('正在初始化数据加载器')
        data = pd.read_csv(str(filedir))

        # 对原始文本进行预处理
        processed_texts = data['text'].apply(
            lambda x: " ".join([w for w in jieba.cut(x) if w not in STOP_SYNCS])
        )

        if tokenizer is None:
            model = models.WordLevel(unk_token="[UNK]")

            self.tokenizer = Tokenizer(model)

            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            trainer = trainers.WordLevelTrainer(
                vocab_size=num_words,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            )

            self.tokenizer.train_from_iterator(processed_texts, trainer=trainer)

            self.tokenizer.save("tokenizer.json")
            debug.success("Tokenizer 初始化完成并已保存")
        else:
            self.tokenizer = tokenizer

        self.tokenizer.enable_padding(
            direction="left",
            length=max_len,
            pad_id=0,
            pad_token="[PAD]"
        )

        # 将文本编码
        encoded_objects = [self.tokenizer.encode(t) for t in processed_texts]
        sequences = [obj.ids for obj in encoded_objects]

        # 循环结束后统一转换成 Tensor
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(data['label'].map(EMOTION_LABELS).values, dtype=torch.long)

        # 词汇表大小
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.max_seq_length = max_len
        self.num_words = num_words

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]