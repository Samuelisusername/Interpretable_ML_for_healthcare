from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os


DATA_PATH = "../../ml4h_data/p2/data/chest_xray/"


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


print(train_dataset.classes, flush=True)       # ['normal', 'pneumonia']
print(train_dataset.class_to_idx, flush=True)  # {'normal': 0, 'pneumonia': 1}


os.makedirs("/data/train_loader")
for i, img in enumerate(train_dataset):
  torch.save(img, '/data/train_loader/train_transformed_img{}'.format(i))

os.makedirs("/data/val_loader")
for i, img in enumerate(val_dataset):
  torch.save(img, '/data/val_loader/val_transformed_img{}'.format(i))

os.makedirs("/data/test_loader")
for i, img in enumerate(test_dataset):
  torch.save(img, '/data/test_loader/test_transformed_img{}'.format(i))