""" src/train.py """
import torch
import torch.optim as optim
from config import (
    DEVICE, STOP_SYNCS,
    EMBEDDING_DIM, HIDDEN_DIM, EMOTION_LABELS,
    EPOCHS, LRS, BATCH_SIZES, MODELS_DIR
)
from src.model import Nya_Nya_ChineseEmotionLSTM
from src.test import test_model
from src.utils import debug, save_records, update_js
from torch.utils.data import DataLoader

def train_model(model, epochs, optim, criterion, train_loader):
    """ epoch训练 """
    prev_loss = None
    for epoch in range(epochs):
        epoch_loss = 0
        for inx, data in enumerate(train_loader):
            inputs, labels = data  # 加载数据
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # 将数据迁移至指定设备
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            optim.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optim.step()  # 更新优化器参数

            # 检查loss波动
            if prev_loss is not None and abs(prev_loss - loss.item()) >= 0.35:
                debug.warning(f'注意: loss波动较大 {prev_loss:.4f} - {loss.item():.4f}')

            if inx % 50 == 0:
                print(f'loss: {loss}')

            prev_loss = loss.item()
            epoch_loss += loss.item()

        # 打印每个epoch的平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f'epoch {epoch + 1}/{epochs}, avg_loss: {avg_loss}')

def train_with_search(train_dataset, test_dataset, criterion, vocab_size):
    """ 超参数搜索中训练 """
    debug.function('正在进行超参数搜索中训练')
    lstm_model = None
    records = []

    for batch_size in BATCH_SIZES:
        for lr in LRS:
            for epoch in EPOCHS:
                # 加载模型
                lstm_model = Nya_Nya_ChineseEmotionLSTM (
                    vocab_size=vocab_size,
                    embedding_dim=EMBEDDING_DIM,
                    hidden_dim=HIDDEN_DIM,
                    num_layers=2
                )
                lstm_model = lstm_model.to(DEVICE)  # 将模型加载至指定设备（GPU/CPU）
                lstm_model.train()  # 训练模式

                # 创建数据加载器
                train_loader = DataLoader(dataset=train_dataset,
                    shuffle=True,  # 打乱
                    pin_memory=True,  # 优化加载速度
                    batch_size=batch_size,  # 批处理大小
                )
                test_loader = DataLoader(dataset=test_dataset,
                    pin_memory=True,
                    batch_size=128,
                )

                # 定义优化器
                optimizer = optim.Adam(lstm_model.parameters(), lr=lr)

                # 训练与测试模型
                train_model(model=lstm_model,
                    epochs=epoch, optim=optimizer,
                    criterion=criterion, train_loader=train_loader
                )
                accuracy = test_model(model=lstm_model, test_loader=test_loader)

                print(f'accuracy: {accuracy * 100:.2f}% | vocab_size: {vocab_size}\n'
                      f' batch_size: {batch_size} | epoch: {epoch} | lr: {lr}')

                # 保存模型
                torch.save({
                    'lr': lr,
                    'epoch': epoch,
                    'batch_size': batch_size,
                    'model_state_dict': lstm_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'vocab_size': vocab_size,
                    'embedding_dim': lstm_model.embedding.embedding_dim,
                    'hidden_dim': lstm_model.lstm.hidden_size,
                    'num_layers': lstm_model.lstm.num_layers,
                }, MODELS_DIR / f'PC_model_acc{accuracy:.4f}.pth')

                debug.success(f'已保存模型文件为: {MODELS_DIR}\\PC_model_acc{accuracy:.4f}.pth')
                update_js('session_state.json', last_trained_model_name=f'PC_model_acc{accuracy:.4f}.pth')

                # 加入记录中便于输出信息与记录
                records.append({
                    'accuracy': accuracy,
                    'batch_size': batch_size,
                    'epoch': epoch,
                    'lr': lr,
                    'vocab_size': vocab_size,
                    'embedding_dim': EMBEDDING_DIM,
                    'hidden_dim': HIDDEN_DIM,
                    'num_words': train_dataset.dataset.num_words,
                    'num_layers': lstm_model.lstm.num_layers,  
                    'num_samples': len(train_dataset),
                })
    # 排序
    records.sort(key=lambda x: x['accuracy'], reverse=True)
    save_records(records, file_name='train_records.txt', mode="train") # 保存记录至文件

    debug.function('训练/保存完成')

    return lstm_model