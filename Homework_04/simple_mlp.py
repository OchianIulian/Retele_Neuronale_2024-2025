import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(28 * 28, 256),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Output layer for 10 classes
        )

    def forward(self, x):
        return self.model(x)
