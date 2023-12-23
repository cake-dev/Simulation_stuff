import torch.nn as nn
import torch.nn.functional as F

class DirectionNet(nn.Module):
    def __init__(self):
        super(DirectionNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # Input layer (3 inputs for x, y, and number of nearby creatures)
        self.fc2 = nn.Linear(10, 4)  # Output layer (4 outputs for up, down, left, right)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=0)  # Use softmax to get a probability distribution
    
    def predict(self, x):
        return self.forward(x).argmax(dim=1)