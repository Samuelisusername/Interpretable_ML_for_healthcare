import os
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. MODEL ARCHITECTURE ---
class CNNClassifier(nn.Module):
    def __init__(self, num_classes, input_size=[384, 384]):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(2 * input_size[0] * input_size[1], num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# --- 2. TRAINING CLASS ---
class CNNTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_loader, num_epochs):
        device = next(self.model.parameters()).device
        self.model.train()
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}] Shuffled Train Loss: {loss.item():.4f}', flush=True)

    def save_weights(self):
        # Save as a distinct file!
        torch.save(self.model.state_dict(), "model_weights_shuffled.pth")
        print("Saved to model_weights_shuffled.pth", flush=True)

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    DATA_PATH = "/home/sagair/ml4h_data/p2/data/chest_xray/chest_xray"
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((384, 384)),               
        transforms.ToTensor()                        
    ])

    print("Loading data and shuffling labels...", flush=True)
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, "train"), transform=transform)
    
    # --- RANDOMIZE LABELS HERE ---
    # Extract labels, shuffle them, and reassign them back to the dataset
    original_labels = [s[1] for s in train_dataset.samples]
    random.shuffle(original_labels)
    train_dataset.samples = [(s[0], l) for s, l in zip(train_dataset.samples, original_labels)]
    train_dataset.targets = original_labels
    # -----------------------------

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = CNNTrainer(model, optimizer, criterion)
    trainer.train(train_loader, num_epochs=10)
    trainer.save_weights()
