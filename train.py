import argparse
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

import numpy as np
import yaml

# custom packages
from utils import *
from load_dataset import *
from Mnist_model import *
from resnet import *

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--params', dest='params')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = params_loaded['gpu_num']

# set seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# set dataset dir
if 'data_dir' not in params_loaded.keys():
    DATA_DIR = './dataset'
else:
    DATA_DIR = params_loaded['data_dir']

# set model dir
MODEL_DIR = "model/{}/{}_{}.pt".format(params_loaded['dataset'],
                                       params_loaded['dataset'],
                                       params_loaded['model_name'])

# directories for experiments
datadir = [DATA_DIR, './model/' + str(params_loaded['dataset'])]
makedirs(datadir)

# set cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# set train/test keyword arguments
kwargs = {'batch_size': params_loaded['batch_size']}
if use_cuda:
    cuda_kwargs = {'num_workers': 2,
                   'pin_memory': True}
    kwargs.update(cuda_kwargs)

# load datasets
train_loader, test_loader = load_data(params_loaded['dataset'], DATA_DIR, kwargs)

# setup model, optimizer, scheduler
model = eval(params_loaded['model_name'])().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=params_loaded['learning_rate'])
scheduler = StepLR(optimizer, step_size=1, gamma=params_loaded['gamma'])

try:
    saved_model = torch.load(MODEL_DIR)
    model = model.load_state_dict(saved_model)
    model.eval()

except:
    for epoch in range(1, params_loaded['epochs'] + 1):
        mnist_train(args, model, device, train_loader, optimizer, epoch)
        mnist_test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), MODEL_DIR)


"""
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
"""