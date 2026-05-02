""" main.py """
import torch
import torch.nn as nn
import sys
import json
import os

from tokenizers import Tokenizer

from config import (
    TRAIN_DATA_DIR,
    EMBEDDING_DIM, HIDDEN_DIM, MODELS_DIR, NUM_WORDS, MAX_LEN,
    STOP_SIGNAL,
)
# 导入数据集加载类和模型类
from src.dataset import SentimentDataSet
from src.inference import self_test
from src.model import load_model

# 导入需要的函数与参数
from src.utils import enter_input, debug, read_js, update_js, get_list_models, save_input_mode
from src.train import train_with_search
from src.colors import blue, red, colors
    
def select_model(lstm_model):
    """ 交互式选择模型 """
    model_list = get_list_models()
    while True:
        js = read_js('session_state.json')

        try:
            # 读取上一次运行的模型并显示这个模型名
            model = load_model(MODELS_DIR / js['model_name'])
            show_name = js['model_name']
        except:
            model = None
            show_name = "未选择"

        print(f'   当前选择的模型: {show_name}')
        print(f'    打印模型列表 -> 输入ls')
        print('    选择模型 -> 输入模型名（models文件夹下）或模型id')
        print('    选择上次使用的模型 -> 输入1')
        print('    选择现有的最佳模型 -> 输入2')
        if js['last_trained_model_name'] is not None:
            print('    选择上次训练的模型 -> 输入3')
        print('    返回上级 -> 输入q')
        print('    进行测试 -> 回车')
        print('->  ', end='')

        selected = input()
        print('-' * 40)

        # 打印模型列表
        if selected == 'ls':
            if model_list == []:
                print('    (空)')
            elif model_list == None:
                print('    未找到目录')
            else:
                for model in model_list:
                    print(f'    {model["id"]} - {model["name"]}')
            print('-' * 40)
            continue

        if selected == '1':
            # 上次使用的模型
            try:
                model = load_model(MODELS_DIR / js['model_name'])  # 读取了模型并修改config
            except FileNotFoundError:
                debug.error('未找到上次使用的模型，请检查models文件夹')
                continue

        elif selected == '2':
            # 选择最佳模型
            model = load_model(MODELS_DIR, find_best=True)
            continue

        elif selected == '3' and js['last_trained_model_name'] is not None:
            # 载入最近训练的模型
            model = lstm_model
            # 更改上次使用的模型为此模型
            update_js('session_state.json', model_name=js['last_trained_model_name'])
            print(f"已加载最近训练的模型")
            continue

        elif selected == '':  # 如果为回车
            if model is not None:
                return model  # 退出函数进入测试
            else:
                debug.error('未选择模型')
                continue  # 返回选择页

        elif selected == 'q' or selected == 'Q':
            return None

        else:  # 如果为输入的路径或索引
            model_path = MODELS_DIR / selected
            # 如果模型存在
            if model_path.exists():
                # 使用自己选择的模型
                model = load_model(model_path)
                continue

            else:  # 如果不存在，试试索引
                found = False
                for m in model_list:
                    if selected == str(m["id"]):
                        selected = m["name"]
                        model_path = MODELS_DIR / selected

                        model = load_model(model_path)
                        found = True
                        break

            if not found:
                debug.error(f"模型文件 {model_path} 不存在")

def interactive_test(model, full_dataset):
    """ 交互式自文本测试 """
    while True:
        js = read_js('session_state.json')  # 保持新的session_state
        input_mode = js["input_mode"]
        mode_text = (
            "普通输入模式" if input_mode == "single" else
            "换行输入模式" if input_mode == "multiline" else
            "列表输入模式" if input_mode == "list" else
            f"{red('暂未选择模式 请输入指令选择')}"
        )
        print(f'当前: {mode_text}')
        print('/change1 - 切换普通输入模式')
        print('/change2 - 切换换行输入模式 允许换行')
        print('/change3 - 切换列表输入模式 允许换行返回列表用于分段测试文本')

        # 自定义输入
        text = enter_input(f"你说: \n{colors['gray']}", stop_signal=STOP_SIGNAL, input_mode=input_mode)

        print(colors['reset'], end='')  # 重置颜色

        # 如果只输入了空字符串则跳出循环
        # 这里因为enter_input的处理逻辑会删掉{STOP_SIGNAL}结束符只剩空字符串
        if text == ['']:
            break

        elif text == None:  # 如果经过enter_input只是修改了输入模式，跳过此次循环重新输入
            continue

        # 测试自己的文本
        for t in text:
            if t.strip():
                self_test(text=t, model=model, tokenizer=full_dataset.tokenizer)


if __name__ == "__main__":
    # 加载或初始化分词器
    tokenizer_path = "tokenizer.json"

    # 查找tokenizer.json文件（词典）是否存在
    if os.path.exists(tokenizer_path):
        current_tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        current_tokenizer = None  # 如果不存在，SentimentDataSet会自动训练并生成

    # 初始化数据集
    full_dataset = SentimentDataSet(
        TRAIN_DATA_DIR,
        tokenizer=current_tokenizer,
        num_words=NUM_WORDS, max_len=MAX_LEN
    )

    # 划分训练集与验证集
    full_dataset_len = len(full_dataset)
    train_dataset_len = int(full_dataset_len * 0.8)
    valid_dataset_len = full_dataset_len - train_dataset_len

    # 分割验证集
    train_dataset, valid_dataset = torch.utils.data.random_split(
        full_dataset, [train_dataset_len, valid_dataset_len],
        generator=torch.Generator().manual_seed(77)
    )

    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    lstm_model = None
    model = None

    # 交互系统
    while True:
        print('选择模式：')
        print('1 - 训练新模型')
        print('2 - 使用现有的模型进行输入文本测试')
        print('q - 退出')
        print('->  ', end='')

        selected_mode = input()
        print('-' * 40)

        # 训练模式
        if selected_mode == '1':
            lstm_model = train_with_search(
                train_dataset=train_dataset, # 使用训练集进行训练
                test_dataset=valid_dataset, # 使用验证集进行测试
                criterion=criterion, # 损失函数
                vocab_size=full_dataset.vocab_size, # 词典大小
            )  # 训练模型
        # 测试模式
        elif selected_mode == '2':
            model = select_model(lstm_model)  # 交互选择模型
            if model:
                interactive_test(model, full_dataset)  # 选择完模型后进行测试

        elif selected_mode == 'q' or selected_mode == 'Q':
            debug.success('程序已退出')
            break  # 跳出循环结束程序

        else:
            debug.error('输入无效')