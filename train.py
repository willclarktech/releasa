import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore
from typing import List


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28_712, 128)
        self.fc2 = nn.Linear(128, 60)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        output = F.log_softmax(x.reshape(x.shape[0], 6, 10), dim=2)
        return output


def stringify_digits(t: torch.Tensor) -> str:
    return "".join([str(n) for n in t.numpy()])


def train(
    model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int,
) -> None:
    model.train()
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output.permute(0, 2, 1), target, reduction="sum")
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{i*len(data)}/{len(train_loader.dataset)} ({100.0 * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}".format()
            )

        if i % 100 == 0:
            target_text = stringify_digits(target[0])
            prediction_text = stringify_digits(output[0].argmax(1))
            print(f"Target: {target_text}; Prediction: {prediction_text}")


def target_transform(folder_names: List[str]):
    def t(i: int) -> torch.Tensor:
        return torch.tensor([int(n) for n in folder_names[i]], dtype=torch.long)

    return t


def main() -> None:
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    datadir = "./data/grayscale-digits-train"
    folder_names = sorted(os.listdir(datadir))
    dataset = datasets.ImageFolder(
        datadir, transform=transform, target_transform=target_transform(folder_names)
    )
    train_loader = DataLoader(dataset, shuffle=True, batch_size=32)

    model = Net()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = 10
    for i in range(n_epochs):
        train(model, train_loader, optimizer, i)

    torch.save(model.state_dict(), "./models/grayscale-digits.pkl")


if __name__ == "__main__":
    main()
