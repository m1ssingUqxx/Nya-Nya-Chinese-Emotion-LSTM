""" src/colors.py """

# ANSI 颜色代码定义
colors = {
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'BLUE': '\033[94m',
    'GRAY': '\033[90m',
    'MAGENTA': '\033[95m',
    'CYAN': '\033[96m',
    'WHITE': '\033[97m',
    'RESET': '\033[0m',

    'yellow': '\033[93m',
    'red': '\033[91m',
    'green': '\033[92m',
    'blue': '\033[94m',
    'gray': '\033[90m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'reset': '\033[0m',
}

# 给文本添加颜色
def red(text):
    return f"{colors['red']}{text}{colors['reset']}"

def green(text):
    return f"{colors['green']}{text}{colors['reset']}"

def yellow(text):
    return f"{colors['yellow']}{text}{colors['reset']}"

def blue(text):
    return f"{colors['blue']}{text}{colors['reset']}"

def magenta(text):
    return f"{colors['magenta']}{text}{colors['reset']}"

def cyan(text):
    return f"{colors['cyan']}{text}{colors['reset']}"

def white(text):
    return f"{colors['white']}{text}{colors['reset']}"
    
def gray(text):
    return f"{colors['gray']}{text}{colors['reset']}"