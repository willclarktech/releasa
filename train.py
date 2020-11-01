import os
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore
from typing import List, Optional


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28712, 128)
        self.fc2 = nn.Linear(128, 60)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
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

        loss = F.nll_loss(
            output.permute(0, 2, 1), target.to(model.device), reduction="sum"
        )
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{i*len(data)}/{len(train_loader.dataset)} ({100.0 * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}".format()
            )
            target_text = stringify_digits(target[0])
            prediction_text = stringify_digits(output[0].cpu().argmax(1))
            print(f"Target: {target_text}; Prediction: {prediction_text}")


def detect_edges(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.FIND_EDGES)


def target_transform(folder_names: List[str]):
    def t(i: int) -> torch.Tensor:
        return torch.tensor([int(n) for n in folder_names[i]], dtype=torch.long)

    return t


counter = 0


def show(img: Image.Image) -> Image.Image:
    counter += 1
    if counter < 5:
        img.show()
    return img


def main(save_path: str, load_path: Optional[str] = None) -> None:
    transform = transforms.Compose(
        [
            detect_edges,
            transforms.Grayscale(),
            # transforms.Resize((50, 20)),
            # show,
            transforms.ToTensor(),
        ]
    )
    datadir = "./data/rgb-digits-train"
    folder_names = sorted(os.listdir(datadir))
    dataset = datasets.ImageFolder(
        datadir, transform=transform, target_transform=target_transform(folder_names)
    )
    train_loader = DataLoader(dataset, shuffle=True, batch_size=32)

    model = Net()
    if load_path:
        model.load_state_dict(torch.load(save_path))
    learning_rate = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = 10
    for i in range(n_epochs):
        train(model, train_loader, optimizer, i)

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    save_path = "./models/rgb-digits.zip"
    load_path = save_path
    main(save_path, load_path)
    # main(save_path)
