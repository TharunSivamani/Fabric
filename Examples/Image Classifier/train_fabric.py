import argparse
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST

# Define Model

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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
        return F.log_softmax(x, dim=1)
    
def run(hparams):

    torch.manual_seed(hparams.seed)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.2,),(0.3,))
        ]
    )

    train_data = MNIST("data/", train=True, download=True,transform=transform)
    test_data = MNIST("data/", train=False, transform=transform)

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=hparams.batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=hparams.batch_size
    )

    model = Net()
    model = model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr = hparams.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

    # Train Loop
    for epoch in range(hparams.epoch):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if(batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                            
            scheduler.step()

            # Testing Loop
            model.eval()
            test_loss = 0
            correct = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    test_loss += F.nll_loss(output, target, reduction="sum").item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            test_loss /= len(test_loader.dataset)

            print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
            )
        )


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    
    hparams = parser.parse_args()
    run(hparams)


if __name__ == "__main__":
    main()