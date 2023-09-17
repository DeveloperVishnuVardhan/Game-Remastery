"""
Author: Jyothi Vishnu Vardhan Kolla
CS-7180 Fall 2023

This file contains the code for training the srcnn model.
"""

from datasets import CustomDataset
from torch.utils.data import DataLoader
from models import SRCNN
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch
from utils import custom_collate


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1296, 1296)),
    # Normalizing using ImageNet stats
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(
    root_dir='data/DIV2K_train_HR/DIV2K_train_HR', transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=4,
                          collate_fn=custom_collate, shuffle=True)

device = 'mps'
model = SRCNN().to(device=device)
criterion = nn.MSELoss()
optimizer = optim.SGD([
    {'params': model.conv1.parameters(), 'lr': 1e-4},
    {'params': model.conv2.parameters(), 'lr': 1e-4},
    {'params': model.conv3.parameters(), 'lr': 1e-5},
], momentum=0.9)

num_epochs = 10

"""
Only save the model if the loss at current epoch is better than 
all the losses in the previous epochs.
"""
best_loss = float('inf')
model.train()
for epoch in range(30):
    train_loss = 0
    for i, (low_res, high_res) in enumerate(train_loader):
        low_res, high_res = low_res.to(device), high_res.to(device)
        output = model(low_res)
        # print(output.shape, high_res.shape)
        loss = criterion(output, high_res)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f"In epoch:{epoch}, step:{i}")

    print(f"Epoch:{epoch}, loss:{loss.item():.4f}")
    if train_loss < best_loss:
        print(f"Saving the best model.")
        torch.save(model.state_dict(), "Models/Srcnn.pth")
