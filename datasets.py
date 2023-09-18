"""
Author: Jyothi Vishnu Vardhan Kolla
CS-7180 Fall 2023

This file contains the code for preparing the dataloaders
for training and validation.
"""

from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """_summary_

        Args:
            root_dir (str): Path to the data 
            transform (_type_, optional): transformations to apply on Images.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(
            root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = os.path.join(self.root_dir, self.image_files[index])
        image = Image.open(image_file).convert("YCbCr")
        y, cb, cr = image.split()

        low_res_img = y.resize(
            (720, 720), Image.ANTIALIAS)
        upscaled_img = low_res_img.resize(
            (y.width, y.height), Image.BICUBIC)

        if self.transform:
            y = self.transform(y)
            upscaled_img = self.transform(upscaled_img)
            # print(upscaled_img.size(), image.size())

        return upscaled_img, y # Returning only the y-channel images.
