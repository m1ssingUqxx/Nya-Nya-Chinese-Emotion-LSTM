""" src/utils.py """
import os
from pathlib import Path

from config import DEVICE, DATA_DIR, MODELS_DIR, RECORDS_DIR, STOP_SIGNAL
from src.colors import colors
import json

class Debugger:
    """ debug工具类 """
    def _log(self, prefix, text, color_key):
        color = colors.get(color_key, colors['RESET'])
        reset = colors['RESET']
        print(f'{color}{prefix}> {text}{reset}')

    def warning(self, text):
        """ 警告 """
        self._log('WARNING', text, 'YELLOW')

    def success(self, text):
        """ 成功 """
        self._log('SUCCESS', text, 'GREEN')

    def error(self, text):
        """ 错误 """
        self._log('ERROR', text, 'RED')

    def model(self, text):
        """ 模型 """
        self._log('MODEL', text, 'MAGENTA')

    def function(self, text):
        """ 函数 """
        self._log('FUNCTION', text, 'BLUE')

debug = Debugger()

def fix_dataset(problematic_indices=None):
    """ 数据集修复函数 """
    debug.function('请稍等...')

    temp_file = DATA_DIR / 'diverse_train_temp.csv'

    with open(DATA_DIR / 'diverse_train.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 修复数据（大小写）
    if problematic_indices is not None:
        # 修复指定的行
        for inx in problematic_indices:
            cols = lines[inx].strip().split(',')
            lines[inx] = f'{cols[0]},{cols[1].lower()}'
    else:
        # # 修复所有行
        fixed_lines = []
        for line in lines:
            cols = line.strip().split(',')
            fixed_lines.append(f'{cols[0]},{cols[1].lower()}')
        lines = fixed_lines

    # 写入临时文件
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # 替换数据集
    os.replace(temp_file, DATA_DIR / 'diverse_train.csv')

    debug.success('数据集已修复完毕')

def save_records(records, file_name, mode):
    """ 保存记录函数 """
    if not(mode == 'train' or mode == 'test'):
        debug.error('模式值错误，请检查save_records函数传入参数')
        return

    if mode == "train":
        for i, record in enumerate(records):
            print('-' * 40)
            print(f'first:{i + 1} | accuracy: {record["accuracy"] * 100:.2f}%')
            print(f'batch_size: {record["batch_size"]} | epoch: {record["epoch"]} | lr: {record["lr"]}')
            print(
                f'vocab_size: {record["vocab_size"]} | embedding_dim: {record["embedding_dim"]} | hidden_dim: {record["hidden_dim"]}')
            print(f'num_words: {record["num_words"]} | num_samples: {record["num_samples"]}\n')

            with open(RECORDS_DIR / file_name, 'a', encoding='utf-8') as f:
                if i == 0:
                    f.write(f"{'-' * 40}\n")
                f.write(f'first:{i + 1} | accuracy: {record["accuracy"] * 100:.2f}%\n')
                f.write(f'batch_size: {record["batch_size"]} | epoch: {record["epoch"]} | lr: {record["lr"]}\n')
                f.write(f'vocab_size: {record["vocab_size"]} | embedding_dim: {record["embedding_dim"]} | hidden_dim: {record["hidden_dim"]}\n')
                f.write(f'num_words: {record["num_words"]} | num_samples: {record["num_samples"]}\n')
                f.write(f"{'-' * 40}\n")
    else:
        for i, record in enumerate(records):
            with open(RECORDS_DIR / file_name, 'a', encoding='utf-8') as f:
                if i == 0:
                    f.write(f"{'-' * 40}\n")
                f.write(f'输入: {record["input"]}\n')
                f.write(f'分词: {record["text_split"]}\n')
                f.write(f'数字序列: {record["sequence"]}\n')
                f.write(f'序列长度: {record["sequence_len"]}\n')
                f.write(f'情感分析:\n')
                for item in record["result"]:
                    f.write(f'  {item["label"]:12} {item["prob"] * 100:5.2f}%\n')
                f.write(f"{'-' * 40}\n")

    with open(RECORDS_DIR / file_name, 'a') as f:
        f.write(f"{'=' * 50}\n\n")

def enter_input(prompt='', stop_signal='/end', show_hint=True, input_mode='multiline'):
    """ 允许换行输入的input函数 """

    def handle_mode_switch(inp):
        """ 检测并处理输入模式切换指令，返回 True 表示已切换 """
        MODE_MAP = {
            '/change1': ('single', '普通输入模式'),
            '/change2': ('multiline', '换行输入模式'),
            '/change3': ('list', '列表输入模式'),
        }

        for cmd, (mode, desc) in MODE_MAP.items():
            if str(inp).strip() == cmd:
                print(f'已切换{desc}')
                update_js('session_state.json', input_mode=mode)
                return True
        return False

    print('只输入结束符可以返回上层')

    if show_hint:
        if input_mode == None:
            print(f'未选择输入模式')
        if input_mode == 'single':
            print(f'此输入模式结束符为 {colors["red"]}换行{colors["reset"]}')
        else:
            end_str = f"{STOP_SIGNAL}"
            print(f'此输入模式结束符为 {colors["red"]}{end_str}{colors["reset"]}')
    print(prompt, end='')

    print('-' * 40)
    if input_mode == 'single': # single模式
        inp = input()
        res = handle_mode_switch(inp)
        if res:
            return
    else: # list或multiline模式
        text_list = []
        while True:
            inp = input()
            res = handle_mode_switch(inp)
            if res:
                return
            # 如果为结束后缀
            if inp.endswith(stop_signal):
                text_list.append(inp[:-len(stop_signal)]) # 切掉结束后缀并append
                break
            else:
                text_list.append(inp)

    if input_mode == 'single': # 如果是普通模式，返回
        return [inp]
    elif input_mode == "list": # 如果是列表模式，直接返回
        return text_list
    elif input_mode == "multiline":
        return ['\n'.join(text_list)] # 列表每个字符串以换行符连接返回

def read_js(file_name):
    """ 读取config工具 """
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)

def update_js(filename, **kwargs):
    """ 更新 config 文件中的任意字段 """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            js = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        js = {}

    js.update(kwargs)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False, indent=4)

def save_model_name(js_name, model_name):
    update_js(js_name, model_name=str(model_name))

def save_input_mode(js_name, input_mode):
    update_js(js_name, input_mode=str(input_mode))

def get_list_models():
    """ 获取models目录下的所有模型文件名 """
    model_list = []
    
    if MODELS_DIR.exists():
        for i, f in enumerate(MODELS_DIR.glob('*.pth')):
            model_list.append({"name": f.name, "id": f"inx{i}"})
        return model_list
    else:
        debug.error(f'未找到模型目录: {MODELS_DIR}')
        return None
