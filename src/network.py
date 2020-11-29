import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, learning_rate: float = 1.0) -> None:
        super(Net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(28712, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 60),
        )
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x.to(self.device))
        return F.log_softmax(output.reshape(output.shape[0], 6, 10), dim=2)
