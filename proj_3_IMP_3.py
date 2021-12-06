from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

import onnx
import onnxruntime as ort

from proj_3_IMP import get_TLoader
from proj_3_IMP_2 import Net


def evaluate(outputs: Variable, labels: Variable) -> float:
    Y = labels.numpy()
    Yhat = np.argmax(outputs, axis=1)
    return float(np.sum(Yhat == Y))


def batch_evaluate(
        net: Net,
        dataloader: torch.utils.data.DataLoader) -> float:
    score = n = 0.0
    for batch in dataloader:
        n += len(batch['image'])
        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs, batch['label'][:, 0])
    return score / n


def validate():
    trainloader, testloader = get_TLoader()
    net = Net().float().eval()

    pretrained_model = torch.load("checkpoint.pth")
    net.load_state_dict(pretrained_model)

    print('=' * 10, 'PyTorch', '=' * 10)
    train_acc = batch_evaluate(net, trainloader) * 100.
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.
    print('Validation accuracy: %.1f' % test_acc)

    trainloader, testloader = get_TLoader(1)

    fname = "signlanguage.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, fname, input_names=['input'])
    model = onnx.load(fname)
    onnx.checker.check_model(model)  
    ort_session = ort.InferenceSession(fname)
    net = lambda inp: ort_session.run(None, {'input': inp.data.numpy()})[0]

    print('=' * 10, 'ONNX', '=' * 10)
    train_acc = batch_evaluate(net, trainloader) * 100.
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.
    print('Validation accuracy: %.1f' % test_acc)


if __name__ == '__main__':
    validate()