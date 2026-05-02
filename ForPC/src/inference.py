""" src/inference.py """
import jieba
import torch
from config import STOP_SYNCS, DEVICE, EMOTION_LABELS, EMOTION_COLORS
from src.dataset import SentimentDataSet
from src.model import Nya_Nya_ChineseEmotionLSTM
from src.utils import debug, save_records
from src.colors import colors, gray

from tokenizers import Tokenizer

def self_test(text, model, tokenizer):
    """ 对输入的文本进行测试 """
    debug.function('文本测试中')
    
    # 分词处理
    words = [x for x in jieba.cut(text, cut_all=False) if x not in STOP_SYNCS]
    text_split = ' '.join(words)

    tokenizer.no_padding()

    # 使用传入的Tokenizer进行编码
    sequence = tokenizer.encode(text_split).ids
    seq_len = len(sequence)

    if seq_len <= 4:
        # 优先处理 如果序列长度为0，则禁用停用词
        if seq_len == 0:
            debug.function('词量为0，禁用停用词')
            text_split = ' '.join([x for x in text])

        # 整个序列都为1，则对每个字进行单独分词
        elif all(idx == 1 for idx in sequence):
            debug.function('未识别到词组, 执行单字重分词')
            text_split = ' '.join([x for x in text if x not in STOP_SYNCS])

        # 整个序列中1占多数，则进行全模式分词
        elif any(idx == 1 for idx in sequence):
            debug.function('词量过小, 执行重分词')
            text_split = ' '.join([x for x in jieba.cut(str(text), cut_all=True) if x not in STOP_SYNCS])

        # 重分词后重新编码
        sequence = tokenizer.encode(text_split).ids
        seq_len = len(sequence)

    # 转换为 Tensor 并移动到指定设备
    inputs = torch.tensor([sequence], dtype=torch.long).to(DEVICE)

    # 模型推理
    with torch.no_grad():
        model.eval()  # 切换到推理模式
        outputs = model(inputs)  # 前向传播

        # 计算概率分布
        prob = torch.softmax(outputs, dim=1)

        # 获取最大概率对应的索引
        pred = torch.argmax(prob, dim=1)[0].item()

        # 获取置信度
        confidence = prob[0][pred]  
        
        # 反向数字标签字典为获取情感打印颜色
        idx_to_label = {v: k for k, v in EMOTION_LABELS.items()}
        sentiment = idx_to_label[pred]

        debug.function('测试完成')

        # 打印信息
        print('-' * 40)
        print(f'输入: \n{gray(f"{text}")}')
        print(f'分词: \n{gray(f"{text_split}")}')
        print(f'数字序列: \n{sequence}')
        print(f'序列长度: {seq_len}')
        print(f'情感分析: {EMOTION_COLORS[sentiment]}{sentiment:2}   {confidence * 100:.2f}%{colors["reset"]}')
        print('其他情感分析：')
        # 格式化打印
        dis = []
        for label, prob_value in zip(EMOTION_LABELS.keys(), prob[0]):
            # 过滤掉概率最大的（已经输出过了）
            if label == sentiment:
                continue
            # 给词典添加这个标签和此标签的概率
            dis.append({'label': label, 'prob': prob_value.item()})
        dis.sort(key=lambda x: x['prob'], reverse=True) # 按概率排序

        # 最后逐行输出
        for item in dis:
            print(f'  {EMOTION_COLORS[item["label"]]}{item["label"]:12} {item["prob"] * 100:5.2f}%{colors["reset"]}')
        print('-' * 40)

        # 加入记录中便于输出信息与记录
        records = [{
            'input': text,
            'text_split': text_split,
            'sequence': sequence,
            'sequence_len': seq_len,
            'result': dis,
        }]

        # 输出结束后，在列表最开始插入概率最大的标签用于保存记录
        dis.insert(0, {'label': sentiment, 'prob': confidence})

        save_records(records, file_name='test_records.txt', mode='test') # 保存
