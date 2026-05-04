import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. ARCHITECTURE (Must match your classifier exactly) ---
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

# --- 2. INTEGRATED GRADIENTS LOGIC ---
def compute_integrated_gradients(image, model, target_class, device, steps=50):
    baseline = torch.zeros_like(image).to(device)
    image = image.to(device)
    
    scaled_inputs = [baseline + (float(i) / steps) * (image - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.cat(scaled_inputs, dim=0)
    scaled_inputs.requires_grad = True
    
    outputs = model(scaled_inputs)
    scores = outputs[:, target_class]
    
    model.zero_grad()
    scores.sum().backward()
    
    grads = scaled_inputs.grad
    avg_grads = torch.mean(grads, dim=0, keepdim=True)
    integrated_grad = (image - baseline) * avg_grads
    
    return integrated_grad.squeeze().cpu().detach().numpy()

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    DATA_PATH = "/home/sagair/ml4h_data/p2/data/chest_xray/chest_xray"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load Model
    model = CNNClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.eval() 
    
    # Load the Data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((384, 384)),               
        transforms.ToTensor()                        
    ])
    test_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, "test"), transform=transform)
    
    print("Searching for 5 healthy and 5 disease samples...", flush=True)
    
    # Collect exactly 5 of each class
    healthy_samples = []  # Class 0
    disease_samples = []  # Class 1
    
    for img, label in test_dataset:
        if label == 0 and len(healthy_samples) < 5:
            healthy_samples.append((img, label))
        elif label == 1 and len(disease_samples) < 5:
            disease_samples.append((img, label))
            
        # Stop searching if we have 5 of both
        if len(healthy_samples) == 5 and len(disease_samples) == 5:
            break

    print("Running Integrated Gradients...", flush=True)
    
    # Create a large plot: 5 rows, 4 columns
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle("Integrated Gradients Attribution Maps", fontsize=20, y=0.98)
    
    for i in range(5):
        # --- Process Healthy (NORMAL) ---
        h_img, h_label = healthy_samples[i]
        h_tensor = h_img.unsqueeze(0)
        h_ig = compute_integrated_gradients(h_tensor, model, target_class=h_label, device=device)
        h_img_np = h_img.squeeze().numpy()
        
        # Plot Healthy Original
        axes[i, 0].imshow(h_img_np, cmap='gray')
        axes[i, 0].set_title(f"Healthy {i+1} - Original")
        axes[i, 0].axis('off')
        
        # Plot Healthy IG Map
        #axes[i, 1].imshow(h_img_np, cmap='gray') removed such that we only have the integrated gradients visualized
        im1 = axes[i, 1].imshow(h_ig, cmap='hot', alpha=0.5)
        axes[i, 1].set_title(f"Healthy {i+1} - IG Map")
        axes[i, 1].axis('off')
        
        # --- Process Disease (PNEUMONIA) ---
        d_img, d_label = disease_samples[i]
        d_tensor = d_img.unsqueeze(0)
        d_ig = compute_integrated_gradients(d_tensor, model, target_class=d_label, device=device)
        d_img_np = d_img.squeeze().numpy()
        
        # Plot Disease Original
        axes[i, 2].imshow(d_img_np, cmap='gray')
        axes[i, 2].set_title(f"Disease {i+1} - Original")
        axes[i, 2].axis('off')
        
        # Plot Disease IG Map
        #axes[i, 3].imshow(d_img_np, cmap='gray') removed for same reason as above.
        im2 = axes[i, 3].imshow(d_ig, cmap='hot', alpha=0.5)
        axes[i, 3].set_title(f"Disease {i+1} - IG Map")
        axes[i, 3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Leave room for the main title
    
    # Save the giant grid
    plt.savefig("ig_10_samples_results.png", bbox_inches='tight')
    print("Saved final grid to ig_10_samples_results.png", flush=True)