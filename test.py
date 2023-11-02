import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
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

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    top1 = 0
    top5 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            _, preds = output.topk(5, 1, True, True)
            correct += (predicted == target).sum().item()
            top1 += (preds[:, :1] == target.view(-1, 1)).sum().item()
            top5 += (preds == target.view(-1, 1)).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)
    top1_accuracy = 100.0 * top1 / len(test_loader.dataset)
    top5_accuracy = 100.0 * top5 / len(test_loader.dataset)
    top1_error = 100.0 - top1_accuracy
    top5_error = 100.0 - top5_accuracy
    return avg_loss, accuracy, top1_error, top5_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ResNet on CIFAR-10 or CIFAR-100')
    parser.add_argument('-dataset', required=True)
    parser.add_argument('-dataset_name', choices=['cifar10', 'cifar100'], required=True)
    parser.add_argument('-model_path', required=True)

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.dataset_name == "cifar10":
        test_dataset = datasets.CIFAR10(root=args.dataset, train=False, transform=transform, download=True)
    else:
        test_dataset = datasets.CIFAR100(root=args.dataset, train=False, transform=transform, download=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_classes=100 if args.dataset_name == 'cifar100' else 10).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    avg_loss, accuracy, top1_error,top5_error = test(model, test_loader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Top1 Error Rate: {top1_error:.2f}%, Top5 Error Rate: {top5_error:.2f}%")
