from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
from utils import progress_bar

import numpy as np
import os
import yaml
import time

from utils import *
from Mnist_model import *
from resnet import *

parser = argparse.ArgumentParser()
parser.add_argument('--params', dest='params')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = params_loaded['gpu_num']

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

datadir = ['dataset', 'model', 'model/' + str(params_loaded['dataset'])]
mkdir(datadir)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': params_loaded['batch_size']}
test_kwargs = {'batch_size': params_loaded['batch_size']}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

if params_loaded['dataset'] == 'mnist':


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST('./dataset', train=True, download=True,
                                    transform=transform)
    test_dataset = datasets.MNIST('./dataset', train=False,
                                    transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # model = eval(params_loaded['model_name'])().to(device)
    model = MnistBaseNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=params_loaded['learning_rate'])

    scheduler = StepLR(optimizer, step_size=1, gamma=params_loaded['gamma'])

    if exists(f'model/mnist/mnist_cnn.pt'):

        model = model.load_state_dict(torch.load(f'./model/mnist/mnist_cnn.pt'))
        model.eval()
    else:

        for epoch in range(1, params_loaded['epochs'] + 1):
            mnist_train(args, model, device, train_loader, optimizer, epoch)
            mnist_test(model, device, test_loader)
            scheduler.step()

        torch.save(model.state_dict(), "model/mnist/mnist_cnn.pt")

elif params_loaded['dataset'] == 'cifar10':

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./dataset', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root='./dataset', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root='./dataset', train=True,download=True,
                                            transform=transform)

    test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True,
                                            transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    net = ResNet50()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('model/cifar-10'):
                os.mkdir('model/cifar-10')
            torch.save(state, './model/cifar-10/cifar10_resnet50.pt')
            best_acc = acc

    if exists(f'model/cifar-10//cifar10_resnet50.pt'):

        print("파일 있다.")
        #model = torch.load(f'model/cifar-10//cifar10_resnet50.pt')

    else:

        for epoch in range(start_epoch, start_epoch+200):
            train(epoch)
            test(epoch)
            scheduler.step()

for data, target in test_loader:
    # print(x.shape)

    data, target = data.to(device), target.to(device)
    output = model(data)
    print(output)
    time.sleep(3)