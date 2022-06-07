from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch

# arguments used for normalization
def norm_param(data_name, get_axis=False):
    """
    주어진 데이터셋에 따른 정규화 파라미터를 출력.

    :param data_name: 데이터셋 이름.
    :param get_axis: 축 정보값 포함 여부를 결정하는 인자.
    """
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

    # set return dictionary
    ret = dict(mean=mean, std=std, axis=axis) if get_axis else dict(mean=mean, std=std)

    return ret


# transform and load datasets
def load_dataset(data_name, data_dir='./dataset', normalize=True):
    """
    주어진 데이터셋과 저장 경로에 따라 train, test dataset을 출력.

    :param data_name: 데이터셋 이름.
    :param data_dir: 데이터셋 저장 경로.
    """
    if normalize:
        preproc = norm_param(data_name)
    else:
        m = (0.0,)
        s = (1.0,)
        if data_name in ['cifar10', 'svhn']:
            m = m * 3
            s = s * 3
        preproc = dict(mean=m, std=s)

    # base transform
    tf_base = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(**preproc)
        ])

    if data_name == 'mnist':
        # load train/test
        train_ds = datasets.MNIST(root=data_dir,
                                  train=True,
                                  download=True,
                                  transform=tf_base)
        test_ds = datasets.MNIST(root=data_dir,
                                 train=False,
                                 download=True,
                                 transform=tf_base)

    elif data_name == 'cifar10':
        # transform for cifar10
        tf_cifar10 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**preproc),
            ])

        # load train/test
        train_ds = datasets.CIFAR10(root=data_dir,
                                    train=True,
                                    download=True,
                                    transform=tf_cifar10)
        test_ds = datasets.CIFAR10(root=data_dir,
                                   train=False,
                                   download=True,
                                   transform=tf_base)

    elif data_name == 'svhn':
        # load train/test
        train_ds = datasets.SVHN(root=data_dir,
                                 split='train',
                                 download=True,
                                 transform=tf_base)
        test_ds = datasets.SVHN(root=data_dir,
                                split='test',
                                download=True,
                                transform=tf_base)

    # no matching dataset name
    else:
        print('error!')
        exit()

    return train_ds, test_ds


# load criterion, optimizer, and scheduler
def load_setup(model_params, data_name, lr):
    """
    데이터셋에 따른 손실함수(criterion), 최적함수(optimizer), 스케쥴러(scheduler) 출력.

    :param model_params: 학습되는 모델의 파라미터.
    :param data_name: 데이터셋 이름.
    :param lr: 학습률 (learning rate).
    """
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


### custom dataset
class CustomDataset(Dataset):
    def __init__(self, xs, ys):
        super().__init__()
        assert len(xs) == len(ys)
        self.x_data = torch.Tensor(xs)
        self.y_data = torch.Tensor(ys)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.Tensor(self.x_data[idx])
        y = torch.LongTensor(self.y_data[idx])
        return x, y