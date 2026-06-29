"""Utilities for running RAFT-Stereo on rectified stereo images."""

import sys, argparse, time
from typing import Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path 

from raft_stereo.raft_stereo import RAFTStereo
from raft_stereo.utils.utils import InputPadder


def load_image(path, device):
    img = np.array(Image.open(path))
    if img.ndim == 2:                      # grayscale -> 3-channel
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:                # drop alpha
        img = img[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def raft_stereo_preproc_img(img: np.ndarray, device) -> torch.Tensor:
    """Image preprocessing for RAFT-Stereo. Converts to 3-channel float tensor."""
    if img.ndim == 2:                      # grayscale -> 3-channel
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:                # drop alpha
        img = img[..., :3]
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)  # (1,3,H,W)
    

def build_raft_model(ckpt_path:Path, extra=None, device=None) -> RAFTStereo:
    """Default RAFT-Stereo architecture flags. Must match training."""
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    p.add_argument("--corr_implementation", default="reg")
    p.add_argument("--shared_backbone", action="store_true")
    p.add_argument("--corr_levels", type=int, default=4)
    p.add_argument("--corr_radius", type=int, default=4)
    p.add_argument("--n_downsample", type=int, default=2)
    p.add_argument("--context_norm", default="batch")
    p.add_argument("--slow_fast_gru", action="store_true")
    p.add_argument("--n_gru_layers", type=int, default=3)
    p.add_argument("--mixed_precision", action="store_true")
    args = p.parse_args(extra or [])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Build RAFT-Stereo model
    model = RAFTStereo(args).to(device).eval()
    # Load pretrained weights
    state = torch.load(ckpt_path, map_location=device)
    # state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    
    return model

def run_raft_stereo(model: RAFTStereo, im0:torch.Tensor, im1:torch.Tensor, iters:int=24)->Tuple[torch.Tensor, float]:
    """Run RAFT-Stereo on a single rectified stereo pair.
    Args:
        model: RAFT-Stereo model
        im0: left image tensor (1,3,H,W)
        im1: right image tensor (1,3,H,W)
        iters: number of iterations to run
    Returns:
        disp: disparity map (H,W)
        runtime: time taken to run (s)
    """
    # Pad images to be divisible by 32
    padder = InputPadder(im0.shape, divis_by=32)
    im0p, im1p = padder.pad(im0, im1)
    t0 = time.time()
    with torch.no_grad():
        _, flow_up = model(im0p, im1p, iters=iters, test_mode=True)
    disp = -padder.unpad(flow_up).squeeze(0).cpu().numpy()
    t1 = time.time()
    return disp, t1-t0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    
    model = build_raft_model(a.ckpt, device=a.device)
    img1 = load_image(a.left, a.device)
    img2 = load_image(a.right, a.device)
    disp, runtime = run_raft_stereo(model, img1, img2, iters=a.iters)
    disp = disp[0] # (1,H,W)
    print(f"Disparity map computed in {runtime:.2f}s, saving to {a.out}")
    lo, hi = np.percentile(disp, [2, 98])
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    Image.fromarray((cm.turbo(norm) * 255).astype(np.uint8)[..., :3]).save(f"{a.out}_disp.png")
    
