import torch
from dataset import get_dataloaders

train_loader, val_loader, classes = get_dataloaders()

images, labels = next(iter(train_loader))

print("Batch Shape: ", images.shape)
print("Labels", labels[:5])
print("Classes: ", classes)