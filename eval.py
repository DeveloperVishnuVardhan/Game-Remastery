"""
Author: Jyothi Vishnu Vardhan Kolla
CS-7180 Fall 2023

This file contains the code for calculating PSNR and SSIM for the model.
"""

import torch
from models import SRCNN
from pytorch_msssim import ssim
import math
from utils import create_test_data_loader


def calculate_PSNR_SSIM(model: torch.nn.Module, criterion: torch.nn, test_loader: torch.utils.data.DataLoader):
    total_psnr = 0
    total_ssim = 0

    # Loop over validation data.
    for i, (low_res, high_res) in enumerate(test_loader):
        low_res, high_res = low_res.to(device), high_res.to(device)
        with torch.no_grad():
            outputs = model(low_res)

        # Calculate PSNR.
        mse = torch.nn.MSELoss()(outputs, high_res)
        psnr = 10 * math.log10(1 / mse.item())
        total_psnr += psnr

        # Calculate SSIM
        ssim_value = ssim(outputs, high_res, data_range=1.0, size_average=True)
        total_ssim += ssim_value.item()

    # Average PSNR and SSIM over validation set
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)

    print(f"Average PSNR on validation set: {avg_psnr}")
    print(f"Average SSIM on validation set: {avg_ssim}")



# Calulate the PSNR and SSIM for SRCNN model.
device = "mps"
SRCNN_model = SRCNN().to(device=device)
SRCNN_model.load_state_dict(torch.load("Models/Srcnn.pth"))
SRCNN_model.eval()
SRCNN_test_loader = create_test_data_loader(
    "data/DIV2K_valid_HR/DIV2K_valid_HR")
criterion = torch.nn.MSELoss()
calculate_PSNR_SSIM(model=SRCNN_model, criterion=criterion,
                    test_loader=SRCNN_test_loader)
