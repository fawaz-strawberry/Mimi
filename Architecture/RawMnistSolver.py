import torch
import torch.nn as nn

class RawMnistSolver(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_outputs)
        self.RELU = nn.ReLU()

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.RELU(x)
        x = self.fc2(x)
        x = self.RELU(x)
        x = self.fc3(x)
        x = self.RELU(x)
        return x