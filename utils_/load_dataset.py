from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.optim as optim
import torch.nn as nn

# dict used for preprocessing
def load_preproc(data_name):
    if data_name == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        axis = -1
    elif data_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        axis = -3
    elif data_name == 'svhn':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        axis = -3
    else:
        print('error!')
        exit()

    return dict(mean=mean, std=std)


# transform and load datasets
def load_data(data_name, data_dir='./dataset', kwargs={}):
    """
    주어진 데이터셋 이름과 경로에 따라 train, test Dataloader 리턴.

    :param data_name: 데이터셋 이름.
    :param data_dir: 데이터셋 경로.
    :param kwargs: 키워드 인자.
    """
    preproc = load_preproc(data_name)

    if data_name == 'mnist':
        # transform
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**preproc)
            ])

        # load train/test
        train_dataset = datasets.MNIST(root=data_dir,
                                       train=True,
                                       download=True,
                                       transform=transform)
        test_dataset = datasets.MNIST(root=data_dir,
                                      train=False,
                                      download=True,
                                      transform=transform)

    elif data_name == 'cifar10':
        # train/test transform
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**preproc),
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**preproc),
            ])

        # load train/test
        train_dataset = datasets.CIFAR10(root=data_dir,
                                         train=True,
                                         download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_dir,
                                        train=False,
                                        download=True,
                                        transform=transform_test)

    elif data_name == 'svhn':
        # transform
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**preproc)
            ])

        # load train/test
        train_dataset = datasets.SVHN(root=data_dir,
                                      split='train',
                                      download=True,
                                      transform=transform)
        test_dataset = datasets.SVHN(root=data_dir,
                                     split='test',
                                     download=True,
                                     transform=transform)

    # no matching dataset name
    else:
        print('error!')
        exit()

    # train/test loader
    train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **kwargs)

    return train_loader, test_loader


# load raw datasets without transformation
def load_raw(data_name, data_dir='./dataset'):
    """
    주어진 데이터셋 이름과 경로에 따라 train, test Dataset 리턴.

    :param data_name: 데이터셋 이름.
    :param data_dir: 데이터셋 경로.
    """
    if data_name == 'mnist':
        # load train/test
        train_raw = datasets.MNIST(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
        test_raw = datasets.MNIST(root=data_dir,
                                  train=False,
                                  download=True,
                                  transform=transforms.ToTensor())

    elif data_name == 'cifar10':
        # load train/test
        train_raw = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     download=True,
                                     transform=transforms.ToTensor())
        test_raw = datasets.CIFAR10(root=data_dir,
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())

    elif data_name == 'svhn':
        # load train/test
        train_raw = datasets.SVHN(root=data_dir,
                                  split='train',
                                  download=True,
                                  transform=transforms.ToTensor())
        test_raw = datasets.SVHN(root=data_dir,
                                 split='test',
                                 download=True,
                                 transform=transforms.ToTensor())

    # no matching dataset name
    else:
        print('error!')
        exit()

    return train_raw, test_raw


# load criterion, optimizer, and scheduler
def load_setup(model_params, data_name, lr):
    if data_name == 'mnist':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model_params, lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    elif data_name in ['cifar10', 'svhn']:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=200)

    else:
        print('error!')
        exit()

    return criterion, optimizer, scheduler