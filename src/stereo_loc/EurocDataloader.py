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
            # Get the disparity map for both frames
            disp0 = self.preprocessor.get_disp_at_timestamp(timestamps0)
            disp0 = torch.from_numpy(disp0).unsqueeze(0).float()
            disp1 = self.preprocessor.get_disp_at_timestamp(timestamps1)
            disp1 = torch.from_numpy(disp1).unsqueeze(0).float()
            # Get the current time and interval between the two frames
            time0 = (
                timestamps0 - self.timestamps[0]
            ) / 1e9  # Convert from nanoseconds to seconds
            time1 = (
                timestamps1 - self.timestamps[0]
            ) / 1e9  # Convert from nanoseconds to seconds
            time_interval = time1 - time0

            return (
                time0,
                time_interval,
                img0_L,
                img1_L,
                disp0,
                disp1,
                T_src_trg,
            )
        except Exception:
            return None


def collate_skip_none(batch):
    # Filter out None entries from the batch
    batch = [item for item in batch if item is not None]

    # If the entire batch became empty, return an empty dict or handle accordingly
    if len(batch) == 0:
        return None

    # Combine the remaining valid samples using standard collate
    return default_collate(batch)


if __name__ == "__main__":
    matplotlib.use("Agg")
    ROOT = Path(__file__).resolve().parents[2]
    # Expected default dataset location inside the experiments tree
    default_root = ROOT / "data" / "Euroc" / "MH_01_easy"
    if not default_root.exists():
        raise FileNotFoundError(f"Euroc dataset not found at {default_root}")

    # get preprocessing object
    euroc_preproc = EurocPreprocess(default_root)
    # get dataset object
    euroc_dataset = EurocDataset(euroc_preproc, frame_interval=1)
    # create a sequential dataloader
    loader = DataLoader(
        euroc_dataset, batch_size=1, shuffle=False, collate_fn=collate_skip_none
    )


    # Create video to check dataloader is working.
    fps = 10
    output_path = str(ROOT / "output_video.avi")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    writer = None

    for i, data in enumerate(loader):
        if i >= 500:
            break
        if data is None:
            continue
        else:
            time0, time_interval, img0_L, img1_L, disp0, disp1, T_src_trg = data
        # Squeeze batch dim; images are uint8 grayscale, disparities are uint16 scaled by 256
        img0 = img0_L[0].numpy()
        img1 = img1_L[0].numpy()
        d0 = disp0[0].float().numpy()
        d1 = disp1[0].float().numpy()

        # Mask zero (invalid) disparity pixels so they don't colour the overlay
        d0_vis = np.where(d0 > 0, d0, np.nan)
        d1_vis = np.where(d1 > 0, d1, np.nan)

        for ax in axes:
            ax.cla()

        axes[0].imshow(img0, cmap="gray", interpolation="nearest")
        axes[0].imshow(d0_vis, cmap="jet", alpha=0.5, interpolation="nearest")
        axes[0].set_title(f"img0  t={float(time0[0]):.2f} s")
        axes[0].axis("off")

        axes[1].imshow(img1, cmap="gray", interpolation="nearest")
        axes[1].imshow(d1_vis, cmap="jet", alpha=0.5, interpolation="nearest")
        axes[1].set_title(f"img1  t={float(time0[0]) + float(time_interval[0]):.2f} s")
        axes[1].axis("off")

        fig.tight_layout(pad=0.5)
        fig.canvas.draw()

        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame_bgr = cv2.cvtColor(buf.reshape(h, w, 4)[:,:,1:], cv2.COLOR_RGB2BGR)

        if writer is None:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                fps,
                (w, h),
            )

        writer.write(frame_bgr)

        if i % 100 == 0:
            print(f"Frame {i}/{len(euroc_dataset)}")

    plt.close(fig)
    if writer is not None:
        writer.release()
    print(f"Video saved to {output_path}")
