import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from typing import Tuple

# Function to load checkpoints of trained networks
def load_checkpoint(
    checkpoint: str, model: nn.Module, learning_rate: float, device: str
):
    model = model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimiser_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss

# Function for sidebranch detection
def predict_sidebranches(
    branch_detector: nn.Module, img_path: str, img_name: str, device: torch.device
) -> Tuple[str, int, np.float32]:
    img = os.path.join(img_path, img_name)
    if not os.path.isfile(img):
        raise IOError("Please enter a valid path.")
    img = np.array(Image.open(img), dtype=np.float32)
    torch_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    torch_img = torch_img.to(device)
    branch_detector.eval()
    prediction = branch_detector(torch_img)
    softmax_pred = F.softmax(input=prediction)
    split_idx = img_name.rfind("_")
    idx = img_name[:split_idx]
    slide_num = int(img_name[split_idx + 1 : -4])

    return idx, slide_num, softmax_pred.detach().cpu().squeeze(0).numpy()
