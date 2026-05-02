""" src/test.py """
import torch

from config import DEVICE
from src.utils import debug

def test_model(model, test_loader):
    """ 对测试集进行测试 """
    debug.model('正在测试模型')
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy