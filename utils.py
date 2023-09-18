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
from PIL import Image

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1296, 1296)),
    # Normalizing using ImageNet stats
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def create_test_data_loader(test_path: str):
    """_summary_

    Args:
        test_path (str): Path of the test dataset.

    Returns:
        _type_: test data loader object.
    """
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

def convert_ycbcr_to_rgb(y, cb, cr):
    """_summary_

    Args:
        y (_type_): Y color space.
        cb (function): Cb color space.
        cr (_type_): Cb color space.

    Returns:
        _type_: _description_
    """
    image = Image.merge("YCbCr", [y, cb, cr]).convert("RGB")
    return image
