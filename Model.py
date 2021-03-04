import torch.nn as nn
import torch.nn.functional as F


class CoolModel(nn.Module):
    def __init__(self, in_features=32, h1=128, h2=512, out_features = 4):
        super().__init__()

        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        # self.fc3 = nn.Linear(h2, h3)
        self.fc3 = nn.Linear(h2, out_features)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    