import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore
from typing import List

from train import Net, target_transform


def main() -> None:
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((50, 20)), transforms.ToTensor(),]
    )
    datadir = "./data/grayscale-digits-test"
    folder_names = sorted(os.listdir(datadir))
    dataset = datasets.ImageFolder(
        datadir, transform=transform, target_transform=target_transform(folder_names)
    )
    batch_size = 1000
    test_loader = DataLoader(dataset, batch_size=batch_size)

    model = Net()
    state_dict = torch.load("./models/grayscale-digits.zip")
    model.load_state_dict(state_dict)
    model.eval()

    score = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions = output.argmax(2, keepdim=True)
            correct = predictions.eq(target.view_as(predictions)).sum(1).flatten()
            score += correct.eq(6).sum().item()

    print(
        f"Correct: {score}/{len(test_loader.dataset)} ({100 * score/len(test_loader.dataset)}%)"
    )


if __name__ == "__main__":
    main()
