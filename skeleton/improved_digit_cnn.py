import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    """
    Simple CNN for MNIST-like (1x28x28) input.
    Architecture (students may improve):
      - conv1 -> bn -> relu -> pool
      - conv2 -> bn -> relu -> pool
      - conv3 -> bn -> relu -> pool
      - fc1 -> relu -> dropout -> fc2
      - log_softmax output
    """
    def __init__(self):
        """
        Construct layers used by the network.
        Students can modify channels, kernel sizes, dropout, etc.
        """
        super().__init__()
        # TODO: Students may modify architecture for better performance
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)   # fewer output channels, larger kernel
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        # Add conv3 to match pooling thrice -> 28->14->7->3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        # after 3 pools spatial dims: 28 -> 14 -> 7 -> 3 (approx)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): input tensor shape (B,1,28,28)

        Returns:
            torch.Tensor: log-probabilities shape (B,10)
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def set_seed(seed=42):
    """
    Set RNG seeds for python, numpy and torch to improve reproducibility.
    Students should call this at the start of training/evaluation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train `model` for a single epoch over `loader`.

    Args:
        model (nn.Module): model to train
        loader (DataLoader): training data loader
        optimizer (torch.optim.Optimizer): optimizer
        criterion: loss function (e.g., nn.CrossEntropyLoss())
        device: torch device

    Returns:
        float: epoch average loss
    """
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        # TODO: Students should implement the training loop:
        #   - move imgs and labels to device
        imgs = imgs.to(device)
        labels = labels.to(device)

        #   - zero optimizer gradients        
        optimizer.zero_grad()

        #   - perform forward pass, compute loss
        logits = model(imgs)        
        loss = criterion(logits, labels)

        #   - backpropagate and step optimizer        
        loss.backward()
        optimizer.step()

        #   - accumulate running_loss
        running_loss += loss.item() * imgs.size(0)
    
    return running_loss / max(1, len(loader.dataset))


def evaluate(model, loader, device):
    """
    Evaluate model accuracy (fraction correct) on given loader.

    Args:
        model (nn.Module): model in eval mode
        loader (DataLoader): dataset to evaluate on
        device: torch device

    Returns:
        float: accuracy in [0.0, 1.0]
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    # Prevent division by zero
    return float(correct / total) if total > 0 else 0.0


def main(args):
    """
    Main entrypoint for training a CNN on MNIST.
    This skeleton sets up datasets, dataloaders, model, loss and optimizer.
    The training loop is present but the model saving step is left as a TODO
    so students implement the persistence behavior themselves.
    """
    # Device selection logic — required by unit tests to be present in source.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # students may adjust num_workers for performance; keep 0 for portability
    num_workers = 0
    train_ds = datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root='.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d}  Train loss: {loss:.4f}  Test Acc: {acc*100:.4f}%")
        if acc > best_acc:
            best_acc = acc
            # TODO: save model here (students implement)
            torch.save(model.state_dict(), 'improved_digit_cnn.pth')

    print(f"Best Test Accuracy during run: {best_acc*100:.4f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MNIST training script (clean version)")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
