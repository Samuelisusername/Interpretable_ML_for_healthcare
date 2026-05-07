import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. ARCHITECTURE (Must match exactly) ---
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

# --- 2. GRAD-CAM LOGIC ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to grab the math inside the layer during forward/backward passes
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Trigger backpropagation for the target class
        score = output[:, target_class]
        score.backward()
        
        # Pool the gradients across the spatial dimensions (Global Average Pooling)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by the gradients
        activations = self.activations.detach().clone().squeeze(0)
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Combine the channels and apply ReLU (we only care about positive influences)
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = torch.relu(heatmap)
        
        # Upsample the heatmap to match the original image size (384x384)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(384, 384), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()
        
        # Normalize between 0 and 1 for plotting
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.cpu().numpy()

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    DATA_PATH = "/home/sagair/ml4h_data/p2/data/chest_xray/chest_xray"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load Model
    model = CNNClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load("Q2/model_weights.pth", map_location=device))
    model.eval() 
    
    # Initialize Grad-CAM on the only convolutional layer we have
    grad_cam = GradCAM(model, model.conv1)
    
    # Load the Data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((384, 384)),               
        transforms.ToTensor()                        
    ])
    test_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, "test"), transform=transform)
    
    print("Searching for 5 healthy and 5 disease samples...", flush=True)
    
    healthy_samples, disease_samples = [], []
    for img, label in test_dataset:
        if label == 0 and len(healthy_samples) < 5:
            healthy_samples.append((img, label))
        elif label == 1 and len(disease_samples) < 5:
            disease_samples.append((img, label))
        if len(healthy_samples) == 5 and len(disease_samples) == 5:
            break

    print("Generating Grad-CAM maps...", flush=True)
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle("Grad-CAM Attribution Maps", fontsize=20, y=0.98)
    
    for i in range(5):
        # Healthy
        h_img, h_label = healthy_samples[i]
        h_tensor = h_img.unsqueeze(0).to(device)
        h_cam = grad_cam.generate(h_tensor, target_class=h_label)
        h_img_np = h_img.squeeze().numpy()
        
        axes[i, 0].imshow(h_img_np, cmap='gray')
        axes[i, 0].set_title(f"Healthy {i+1} - Original")
        axes[i, 0].axis('off')
        
        #axes[i, 1].imshow(h_img_np, cmap='gray') removed because we only want the Grad-CAM not overlayed by the original
        axes[i, 1].imshow(h_cam, cmap='jet', alpha=0.5) # Using 'jet' for classic Grad-CAM look
        axes[i, 1].set_title(f"Healthy {i+1} - Grad-CAM")
        axes[i, 1].axis('off')
        
        # Disease
        d_img, d_label = disease_samples[i]
        d_tensor = d_img.unsqueeze(0).to(device)
        d_cam = grad_cam.generate(d_tensor, target_class=d_label)
        d_img_np = d_img.squeeze().numpy()
        
        axes[i, 2].imshow(d_img_np, cmap='gray')
        axes[i, 2].set_title(f"Disease {i+1} - Original")
        axes[i, 2].axis('off')
        
        #axes[i, 3].imshow(d_img_np, cmap='gray') removed for the same reason as above
        axes[i, 3].imshow(d_cam, cmap='jet', alpha=0.5)
        axes[i, 3].set_title(f"Disease {i+1} - Grad-CAM")
        axes[i, 3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("gradcam_10_samples_results.png", bbox_inches='tight')
    print("Saved final grid to gradcam_10_samples_results.png", flush=True)
