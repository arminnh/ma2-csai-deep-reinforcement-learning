import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, possible_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pre_head = nn.Linear(64*7*7, 512)
        self.head = nn.Linear(512, possible_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.pre_head(x.view(x.size(0), -1)))
        return self.head(x)
