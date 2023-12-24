import torch.nn as nn
import torch.nn.functional as F

class DirectionNet(nn.Module):
    def __init__(self):
        super(DirectionNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.bn1 = nn.InstanceNorm1d(10)
        self.fc2 = nn.Linear(10, 4)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add an extra dimension for batch size
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x + 1e-8, dim=1)  # Change dim to 1 as we added an extra dimension
    
    def predict(self, x):
        x = x.unsqueeze(0)  # Add an extra dimension for batch size
        return self.forward(x).argmax(dim=1)