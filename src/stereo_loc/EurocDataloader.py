from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch
from pathlib import Path
import matplotlib

import matplotlib.pyplot as plt
import cv2
import numpy as np

from stereo_loc.EurocPreprocess import EurocPreprocess


class EurocDataset(Dataset):
    def __init__(self, preprocessor: EurocPreprocess, frame_interval: int = 1):
        self.preprocessor: EurocPreprocess = preprocessor
        self.timestamps = preprocessor.cam0.timestamps
        self.frame_interval = frame_interval
        # Find first index

    def __len__(self):
        # Valid length is the number of timestamps minus the frame skip, since we can't use the last few frames if we're skipping.
        return len(self.timestamps) - self.frame_interval

    def __getitem__(self, idx):
        try:
            timestamps0 = self.timestamps[idx]
            timestamps1 = self.timestamps[idx + self.frame_interval]
            # Get relative transformation between the two frames
            T_src_trg = self.preprocessor.get_relative_transform(
                timestamps0, timestamps1, camera_frame=True
            )
            # Get the images for the two frames
            img0_L, _ = self.preprocessor.get_image_at_timestamp(
                timestamps0, rectify=True
            )
            img1_L, _ = self.preprocessor.get_image_at_timestamp(
                timestamps1, rectify=True
            )
            img0_L = process_image(img0_L)
            img1_L = process_image(img1_L)
            # Get the disparity map for both frames
            disp0 = self.preprocessor.get_disp_at_timestamp(timestamps0)
            disp1 = self.preprocessor.get_disp_at_timestamp(timestamps1)
            # Get the current time and interval between the two frames
            time0 = (
                timestamps0 - self.timestamps[0]
            ) / 1e9  # Convert from nanoseconds to seconds
            time1 = (
                timestamps1 - self.timestamps[0]
            ) / 1e9  # Convert from nanoseconds to seconds
            time_interval = time1 - time0

            return (
                idx,
                time0,
                time_interval,
                img0_L,
                img1_L,
                disp0,
                disp1,
                T_src_trg,
            )
        except Exception as e:
            print(f"Error processing index {idx}. Skipping this sample.")
            print(f"Exception:\n {e}")
            return None


def process_image(img: np.ndarray) -> np.ndarray:
    """Convert image to float32 and normalize to [0, 1]. If the image has multiple channels, convert to grayscale.
    Also add batch dimension to make it (1, H, W) for compatibility with the model."""
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        # convert to grayscale by averaging channels
        img = img.mean(axis=2)
    return img


def collate_skip_none(batch):
    # Filter out None entries from the batch
    batch = [item for item in batch if item is not None]

    # If the entire batch became empty, return an empty dict or handle accordingly
    if len(batch) == 0:
        return None

    # Combine the remaining valid samples using standard collate
    return default_collate(batch)
