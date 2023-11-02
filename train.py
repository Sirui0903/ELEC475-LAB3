import argparse
import torch
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from resnet import *


class Net(nn.Module):
    def __init__(self,num_classes):
        super(Net, self).__init__()
        self.encoder_decoder = encoder_decoder.encoder
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder_decoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100 * correct / len(train_loader.dataset)
    return avg_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10 or CIFAR-100')

    parser.add_argument('-dataset', required=True, help='Path to CIFAR dataset folder')
    parser.add_argument('-dataset_name', choices=['cifar10', 'cifar100'], required=True,
                        help='Choose between CIFAR-10 or CIFAR-100')
    parser.add_argument('-output', required=True, help='Name of the output .pth file')
    parser.add_argument('-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-batchsize', type=int, default=128, help='Training batch size')
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate for the optimizer')

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=args.dataset, train=True, transform=transform, download=True)
    else:
        train_dataset = datasets.CIFAR100(root=args.dataset, train=True, transform=transform, download=True)


    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10 if args.dataset_name == "cifar10" else 100
    model = Net(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    accuracies = []

    for epoch in range(args.epochs):
        avg_loss, accuracy = train(model, train_loader, criterion, optimizer, device)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1} :       Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")
    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, '-b', label='Training Loss')
    plt.plot(epochs, accuracies, '-r', label='Training Accuracy (%)')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"./{args.output}.png")
    plt.show()
