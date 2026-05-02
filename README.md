# 中文情感分析模型(LSTM + Attention)

基于 PyTorch，支持 7 种细粒度情感分类：<br>
**love&nbsp;&nbsp;neutral&nbsp;&nbsp;sadness&nbsp;&nbsp;anxiety&nbsp;&nbsp;joy&nbsp;&nbsp;contentment&nbsp;&nbsp;anger**

> **数据来源声明**
> 
> 本项目使用的所有训练数据均由大语言模型生成。
> 
> - **用途限制**：仅供个人学习、学术研究及技术验证
> - **已知问题**：AI 生成数据可能存在情感表达单一、语义重复、与文化语境脱节等问题
> 
> **关于预测结果**
> 
> 模型输出的是"哪项情绪最有可能"，数值仅代表模型对这一判断的置信度，**并非各项情绪的占比**

## 版本说明
本项目提供两个运行版本：
- **[PC 版](ForPC)**：标准Python环境，命令行交互
- **[移动版](ForMobile)**：替换了 `tokenizers` 为纯 Python 实现，解决 Termux/Pydroid3 等移动端环境下无法编译 Rust 扩展的问题。核心功能与 PC 版一致。

## 项目结构
```
Nya-Nya-Chinese-Emotion-LSTM/  总目录
├── data/                      训练数据
├── records/                   训练/测试记录
├── ForPC/                     PC端入口
│   ├── main.py                启动文件
│   ├── config.py              配置文件
│   ├── models/                模型权重
│   ├── src/                   核心代码
│   ├── tokenizer.json         分词器词表
│   └── session_state.json     上下文管理
│ 
├── ForMobile/                 移动端入口（结构同上PC端）
```
> **注意**：PC端与移动端使用的分词器不同导致词表不通用，需各自训练。

## 你需要哪些外部库？
### ForPC
- `torch` — 深度学习框架
- `tokenizers` — 分词器
- `jieba` — 中文分词
- `pandas` — 数据处理
### ForMobile
- `torch` — 深度学习框架
- `jieba` — 中文分词
- `pandas` — 数据处理

```bash
pip install torch
pip install tokenizers
pip install jieba
pip install pandas
```


## 快速选择
| 你的环境              | 进入                        |
|-------------------|-------------------------------|
| Windows/Linux/Mac | [ForPC](forPC) |
| Android/树莓派/终端设备  | [ForMobile](forMobile)|

## 开始
**PC端**
```bash
cd ForPC
python main.py
```
**移动端**
```bash
cd ForMobile
python main.py
```
---
