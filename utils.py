"""
Author: Jyothi Vishnu Vardhan Kolla
CS-7180 Fall 2023

This file contains the code for utility functions used in the project.
"""

from datasets import CustomDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.nn.functional import pad

transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1296, 1296)),
        # Normalizing using ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

def create_test_data_loader(test_path: str):
    test_dataset = CustomDataset(
        root_dir=test_path, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=4,
                             collate_fn=custom_collate, shuffle=True)
    return test_loader


def custom_collate(batch):
    max_size = max([x[0].size(1) for x in batch])
    padded_batch = []
    for (img, target) in batch:
        pad_amount = max_size - img.size(1)
        padded_img = pad(img, (0, 0, 0, pad_amount))
        padded_batch.append((padded_img, target))
    return default_collate(padded_batch)
