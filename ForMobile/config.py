""" config.py """
import torch

from pathlib import Path
from src.colors import colors

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'data'
MODELS_DIR = BASE_DIR / 'models'
RECORDS_DIR = BASE_DIR / 'records'
TRAIN_DATA_DIR = DATA_DIR / 'diverse_train.csv'

# 每个类别的编码
EMOTION_LABELS = {
    'joy': 0, 'sadness': 1,
    'anger': 2, 'love': 3,
    'anxiety': 4, 'contentment': 5,
    'neutral': 6
}
# 每个类别输出的颜色
EMOTION_COLORS = {
    'joy': colors['green'],
    'sadness': colors['blue'],
    'anger': colors['red'],
    'love': colors['magenta'],
    'anxiety': colors['yellow'],
    'contentment': colors['cyan'],
    'neutral': colors['gray'],
}

# 定义停用词/符号
STOP_SYNCS = [
    "，", "。", "！", "？", "“", "”", "‘", "’", "（", "）", "【", "】", "《",
    "》", "；", "：", "、", "…", "—", "～", "#", "@", " ", "\t"
]
STOP_SYNCS += [
    "的", "地", "得", "了", "着", "过", "所", "之", "等等", "的话",
    "呢", "呗", "哈", "噢", "哇", "是",
    "在", "于", "从", "自", "往", "向", "到", "把", "被", "让", "对", "对于",
    "关于", "根据", "按照", "经过", "除了", "随着", "当", "以",
    "与", "及", "以及", "或", "或者", "并且", "而且", "另外", "同时",
    "此外", "因此", "所以", "于是", "然后", "接着", "从而", "因而",
    "因为", "既然", "例如", "比如", "比方", "其中",
    "你们", "他们", "她们",
    "这", "那", "这个", "那个", "这些", "那些", "这里", "那里", "某", "各",
    "个", "些", "种", "次", "位", "点", "件", "条", "每", "各", "该",
]

MAX_LEN = 50 # 单条样本的最大序列长度
EMBEDDING_DIM = 256 # 词嵌入维度
HIDDEN_DIM = 256 # LSTM隐层维度

# 最大词表量，超出部分截掉以减少噪音
NUM_WORDS = 10000 # 若要应用修改需要删掉tokenizer.json词表，但旧模型将因 索引映射冲突 而失效

# 超参数
EPOCHS = [25]
LRS = [0.002, 0.00025]
BATCH_SIZES = [128]

# 长输入语句的结束符
STOP_SIGNAL = '/end'

