import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistBaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MaskedMnistBaseNet(MnistBaseNet):
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
        x = self.conv2(x)
        x = self.apply_mask(x, self.masks['conv2'])
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.apply_mask(x, self.masks['dropout1'])
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.apply_mask(x, self.masks['fc1'])
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.apply_mask(x, self.masks['dropout2'])
        x = self.fc2(x)
        x = self.apply_mask(x, self.masks['fc2'])
        output = F.log_softmax(x, dim=1)
        return output