from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import DataLoader

class CNNClassifier(nn.Module):

    def __init__(self, num_classes, input_size=[384, 384]):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(8 * input_size[0] * input_size[1], num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    
class CNNTrainer:

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_loader, val_loader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                val_loss = 0
                for images, labels in val_loader:
                    outputs = self.model(images)
                    val_loss += self.criterion(outputs, labels).item()
                val_loss /= len(val_loader)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
    
    def evaluate(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            total = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.size(0)
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                tn += ((predicted == 0) & (labels == 0)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()

        print(f'Accuracy: {100 * tp / total:.2f}%')
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

        auroc = roc_auc_score(labels.numpy(), outputs.numpy())
        print(f'AUROC: {auroc:.4f}')
        auprc = average_precision_score(labels.numpy(), outputs.numpy())
        print(f'AUPRC: {auprc:.4f}')

if __name__ == "__main__":
    DATA_PATH = "../ml4h_data/p2/data/chest_xray/"

    # Step 1 — compute mean and std from training set (grayscale, so 1 channel)
    tmp_dataset = datasets.ImageFolder(root=DATA_PATH + "train", transform=transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Grayscale(),
        transforms.ToTensor(),  # scales to [0, 1]
    ]))

    print("hello", flush=True)
    loader = DataLoader(tmp_dataset, batch_size=64, shuffle=False)

    mean, std, n = 0.0, 0.0, 0
    for images, _ in loader:
        mean += images.mean()
        std  += images.std()
        n    += 1

    mean /= n
    std  /= n
    print(f"Mean: {mean:.4f}, Std: {std:.4f}", flush=True)

    # Step 2 — apply normalization
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),  # 1 channel
    ])



    train_dataset = datasets.ImageFolder(root=DATA_PATH + "train", transform=transform)
    val_dataset   = datasets.ImageFolder(root=DATA_PATH + "val",   transform=transform)
    test_dataset  = datasets.ImageFolder(root=DATA_PATH + "test",  transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Example usage
    num_classes = 2
    model = CNNClassifier(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = CNNTrainer(model, optimizer, criterion)

    # Assuming train_loader, val_loader, and test_loader are defined
    trainer.train(train_loader, val_loader, num_epochs=10)
    trainer.evaluate(test_loader)