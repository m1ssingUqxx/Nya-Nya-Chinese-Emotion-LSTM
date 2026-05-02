"""
    src/my_tokenizers.py
    此py文件为AI编写 目的为在手机环境模拟使用Tokenizers库
"""

import json

class WordLevel:
    def __init__(self, vocab=None, unk_token="[UNK]"):
        self.vocab = vocab or {}
        self.unk_token = unk_token

class WordLevelTrainer:
    def __init__(self, vocab_size=5000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []

class Encoding:
    def __init__(self, ids):
        self.ids = ids

class Tokenizer:
    def __init__(self, model):
        self.model = model
        self.vocab = model.vocab

    def enable_padding(self, direction="right", pad_id=0, pad_token="[PAD]", length=None):
        self.padding_params = {
            "direction": direction,
            "pad_id": pad_id,
            "length": length
        }

    def get_vocab_size(self, with_added_tokens=True):
        # 直接返回字典里的键值对数量
        return len(self.vocab)

    def train_from_iterator(self, iterator, trainer):
        # 简单的频率统计模拟训练
        words = {}
        for text in iterator:
            for word in text.split():
                words[word] = words.get(word, 0) + 1

        # 排序并取前 vocab_size 个
        sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)

        # 构建词表：先放特殊符号
        new_vocab = {token: i for i, token in enumerate(trainer.special_tokens)}
        for word, _ in sorted_words:
            if len(new_vocab) >= trainer.vocab_size: break
            if word not in new_vocab:
                new_vocab[word] = len(new_vocab)

        self.vocab = new_vocab
        self.model.vocab = new_vocab

    def encode(self, text):
        tokens = text.split()
        ids = [self.vocab.get(t, self.vocab.get("[UNK]", 1)) for t in tokens]
        return Encoding(ids)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"model": {"vocab": self.vocab}}, f)

    def get_vocab(self):
        return self.vocab

    @classmethod
    def from_file(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(WordLevel(vocab=data['model']['vocab']))