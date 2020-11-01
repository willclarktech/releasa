#!/usr/bin/env python3
import os
from PIL import Image, ImageFilter  # type: ignore
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore
from typing import List, Optional

from network import Net
from utils import detect_edges, stringify_digits, target_transform


def train(model: Net, train_loader: DataLoader, epoch: int,) -> None:
    model.train()
    for i, (data, target) in enumerate(train_loader):
        model.optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(
            output.permute(0, 2, 1), target.to(model.device), reduction="sum"
        )
        loss.backward()
        model.optimizer.step()

        if i % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{i*len(data)}/{len(train_loader.dataset)} ({100.0 * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}".format()
            )
            target_text = stringify_digits(target[0])
            prediction_text = stringify_digits(output[0].cpu().argmax(1))
            print(f"Target: {target_text}; Prediction: {prediction_text}")


def main(
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    datadir: str,
    save_path: str,
    load_path: Optional[str] = None,
) -> None:
    transform = transforms.Compose(
        [detect_edges, transforms.Grayscale(), transforms.ToTensor(),]
    )
    folder_names = sorted(os.listdir(datadir))
    dataset = datasets.ImageFolder(
        datadir, transform=transform, target_transform=target_transform(folder_names)
    )
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    model = Net(learning_rate)
    if load_path:
        model.load_state_dict(torch.load(save_path))

    for i in range(n_epochs):
        train(model, train_loader, i)

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    n_epochs = 10
    batch_size = 32
    learning_rate = 1e-5
    datadir = "./data/rgb-digits-train"
    save_path = "./models/rgb-digits.zip"
    load_path = save_path
    # main(n_epochs, batch_size, learning_rate, datadir, save_path, load_path)
    main(n_epochs, batch_size, learning_rate, datadir, save_path)
