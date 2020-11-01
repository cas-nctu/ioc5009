import os
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    #print("{'Param name':20} {'Shape':30} {'Type':15}")
    #print('-'*70)
    for name, param in model.named_parameters():
        print('name:', name, 'param.shape:', param.shape, 'param.dtype', param.dtype)
        if with_values:
            print(param)


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print('name:', name, ' nonzeros = ', nz_count, 'params:', total_params, 'pruned =', (total_params - nz_count))
    print('alive:', nonzero , 'pruned :', total - nonzero , 'total:', total, 'Compression Ratio:', ((total-nonzero)/total))


def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print('Test set: Average loss:%.4f, Accuracy:%.2f)' %(test_loss, accuracy))
    return accuracy
