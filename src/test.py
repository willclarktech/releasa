#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore
from typing import List

from network import Net
from utils import detect_edges, target_transform


def main(datadir: str, load_path: str) -> None:
    transform = transforms.Compose(
        [detect_edges, transforms.Grayscale(), transforms.ToTensor(),]
    )
    folder_names = sorted(os.listdir(datadir))
    dataset = datasets.ImageFolder(
        datadir, transform=transform, target_transform=target_transform(folder_names)
    )
    batch_size = len(folder_names)
    test_loader = DataLoader(dataset, batch_size=batch_size)

    model = Net()
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    model.eval()

    score = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions = output.cpu().argmax(2, keepdim=True)
            correct = predictions.eq(target.view_as(predictions)).sum(1).flatten()
            score += correct.eq(6).sum().item()

    print(
        f"Correct: {score}/{len(test_loader.dataset)} ({100 * score/len(test_loader.dataset)}%)"
    )


if __name__ == "__main__":
    datadir = "./data/rgb-digits-test"
    load_path = "./models/rgb-digits.zip"
    main(datadir, load_path)
