'''LeNet in PyTorch, for MNIST'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)


# masked LeNet
class MaskedLeNet(LeNet):
    def __init__(self, masks=None):
        super().__init__()
        self.masks = masks

    def apply_mask(self, x, m):
        device = x.get_device()
        b = x.size(0)                           # batch size
        d = tuple(1 for _ in range(x.dim()-1))
        ext_m = m.repeat(b,*d).to(device)       # extended mask
        return torch.where(ext_m, x, torch.zeros_like(x))

    # TODO: automate this process?
    def forward(self, x):
        x = self.conv1(x)
        x = self.apply_mask(x, self.masks['conv1'])
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = self.apply_mask(x, self.masks['conv2'])
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv3(x)
        x = self.apply_mask(x, self.masks['conv3'])
        x = F.relu(x)
        x = self.flatten(x)
        x = self.apply_mask(x, self.masks['flatten'])
        x = self.fc1(x)
        x = self.apply_mask(x, self.masks['fc1'])
        x = F.relu(x)
        x = self.fc2(x)
        x = self.apply_mask(x, self.masks['fc2'])
        output = F.log_softmax(x, dim=1)
        return output