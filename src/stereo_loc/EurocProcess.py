from __future__ import annotations

from dataclasses import dataclass
from typing import List
from pathlib import Path
from typing import Any
import csv

import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation, RigidTransform
import cv2

from utils.stereo_camera_model import get_disparity, StereoCameraConfig


@dataclass(frozen=True)
class CameraSensorInfo:
    sensor_type: str | None
    comment: str | None
    T_bs: np.ndarray
    rate_hz: float | None
    resolution: tuple[int, int] | None
    camera_model: str | None
    intrinsics: np.ndarray
    distortion_model: str | None
    distortion_coefficients: np.ndarray

@dataclass(frozen=True)
class CameraInfo:
    path: Path
    sensor: CameraSensorInfo
    data_dir: Path
    data_csv: Path
    timestamp_to_file: dict[int, Path]
    timestamps: list[int]


@dataclass(frozen=True)
class GroundtruthInfo:
    path: Path
    sensor: dict[str, Any] | None
    data_csv: Path


@dataclass(frozen=True)
class GroundtruthData:
    timestamps_to_index: dict[int, int]
    timestamps: np.ndarray
    p_rs_r: np.ndarray
    T_rs: np.ndarray
    v_rs_r: np.ndarray
    b_w_rs_s: np.ndarray
    b_a_rs_s: np.ndarray


@dataclass(frozen=True)
class StereoCamera:
    camera_fx: float
    camera_fy: float
    camera_cx: float
    camera_cy: float
    camera_k1: float
    camera_k2: float
    camera_p1: float
    camera_p2: float
    camera_width: int
    camera_height: int
    camera_fps: float
    camera_bf: float
    camera_rgb: int
    th_depth: float
    left_height: int
    left_width: int
    left_d: np.ndarray
    left_k: np.ndarray
    left_r: np.ndarray
    left_p: np.ndarray
    right_height: int
    right_width: int
    right_d: np.ndarray
    right_k: np.ndarray
    right_r: np.ndarray
    right_p: np.ndarray


class EurocDataset:
    def __init__(
        self,
        path: Path,
        stereo_params: Path = Path(
            "/workspace/experiments/data/Euroc/EurocStereo.yaml"
        ),
    ):
        self.root = Path(path)
        if not self.root.exists():
            raise FileNotFoundError(f"Euroc dataset path not found: {self.root}")

        seq_dir = self._resolve_sequence_root(self.root)
        mav0_dir = seq_dir / "mav0"
        cam0_dir = mav0_dir / "cam0"
        cam1_dir = mav0_dir / "cam1"

        self.path = seq_dir
        self.mav0_path = mav0_dir
        self.body = self._load_yaml_if_exists(mav0_dir / "body.yaml")
        self.cam0 = self._load_camera(cam0_dir, "cam0")
        self.cam1 = self._load_camera(cam1_dir, "cam1")
        self.gt_info: GroundtruthInfo = self._load_groundtruth(
            mav0_dir / "state_groundtruth_estimate0"
        )
        self.gt_data: GroundtruthData = self.process_groundtruth()
        self.stereo_camera: StereoCamera = self._load_stereo_camera(stereo_params)

        # Initialize stereo rectification maps
        self.cam0_rect_map = cv2.initUndistortRectifyMap(
            self.stereo_camera.left_k,
            self.stereo_camera.left_d,
            self.stereo_camera.left_r,
            self.stereo_camera.left_p[:3, :3],
            (
                self.stereo_camera.left_width,
                self.stereo_camera.left_height,
            ),
            cv2.CV_32F,
        )
        self.cam1_rect_map = cv2.initUndistortRectifyMap(
            self.stereo_camera.right_k,
            self.stereo_camera.right_d,
            self.stereo_camera.right_r,
            self.stereo_camera.right_p[:3, :3],
            (
                self.stereo_camera.right_width,
                self.stereo_camera.right_height,
            ),
            cv2.CV_32F,
        )

    def _resolve_sequence_root(self, root: Path) -> Path:
        if (root / "mav0").exists():
            return root

        candidates: list[Path] = []
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            if seq_dir.name.startswith("."):
                continue
            if (seq_dir / "mav0").exists():
                candidates.append(seq_dir)

        if not candidates:
            raise FileNotFoundError(f"No Euroc sequences found under: {root}")

        if len(candidates) > 1:
            names = ", ".join(seq.name for seq in candidates)
            raise FileNotFoundError(
                f"Multiple Euroc sequences found under {root}: {names}. "
                "Provide a single sequence root."
            )

        return candidates[0]

    def _load_camera(self, cam_dir: Path, name: str) -> CameraInfo:
        if not cam_dir.exists():
            raise FileNotFoundError(f"{name} directory not found: {cam_dir}")

        sensor_yaml = cam_dir / "sensor.yaml"
        if not sensor_yaml.exists():
            raise FileNotFoundError(f"{name} sensor.yaml not found: {sensor_yaml}")

        data_csv = cam_dir / "data.csv"
        if not data_csv.exists():
            raise FileNotFoundError(f"{name} data.csv not found: {data_csv}")
        timestamp_to_file = self._load_camera_mapping(data_csv, cam_dir / "data")
        timestamps = list(timestamp_to_file.keys())

        return CameraInfo(
            path=cam_dir,
            sensor=self._parse_camera_sensor(self._load_yaml(sensor_yaml)),
            data_dir=cam_dir / "data",
            data_csv=data_csv,
            timestamp_to_file=timestamp_to_file,
            timestamps=timestamps,
        )

    def _load_camera_mapping(self, data_csv: Path, data_dir: Path) -> dict[int, Path]:
        mapping: dict[int, Path] = {}
        with data_csv.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue
                timestamp = int(row[0])
                filename = row[1]
                mapping[timestamp] = data_dir / filename
        return mapping

    def _parse_camera_sensor(self, sensor: dict[str, Any]) -> CameraSensorInfo:
        t_bs_data = sensor.get("T_BS", {})
        rows = int(t_bs_data.get("rows", 0))
        cols = int(t_bs_data.get("cols", 0))
        data = t_bs_data.get("data", [])
        if rows > 0 and cols > 0 and data:
            t_bs = np.array(data, dtype=float).reshape(rows, cols)
        else:
            t_bs = np.empty((0, 0), dtype=float)

        resolution = sensor.get("resolution")
        if resolution is not None:
            resolution = (int(resolution[0]), int(resolution[1]))

        intrinsics = sensor.get("intrinsics")
        intrinsics_array = (
            np.array(intrinsics, dtype=float)
            if intrinsics is not None
            else np.array([])
        )

        distortion_coefficients = sensor.get("distortion_coefficients")
        distortion_array = (
            np.array(distortion_coefficients, dtype=float)
            if distortion_coefficients is not None
            else np.array([])
        )

        rate_hz = sensor.get("rate_hz")
        return CameraSensorInfo(
            sensor_type=sensor.get("sensor_type"),
            comment=sensor.get("comment"),
            T_bs=t_bs,
            rate_hz=float(rate_hz) if rate_hz is not None else None,
            resolution=resolution,
            camera_model=sensor.get("camera_model"),
            intrinsics=intrinsics_array,
            distortion_model=sensor.get("distortion_model"),
            distortion_coefficients=distortion_array,
        )

    def _load_groundtruth(self, gt_dir: Path) -> GroundtruthInfo | None:
        if not gt_dir.exists():
            return None

        sensor_yaml = gt_dir / "sensor.yaml"
        return GroundtruthInfo(
            path=gt_dir,
            sensor=self._load_yaml_if_exists(sensor_yaml),
            data_csv=gt_dir / "data.csv",
        )

    def _load_stereo_camera(self, stereo_params: Path) -> StereoCamera:
        """Loads data from the stereo rectification YAML file.
        Expected file should match the one used for ORBSLAM"""
        if not stereo_params.exists():
            raise FileNotFoundError(f"Stereo params not found: {stereo_params}")

        fs = cv2.FileStorage(str(stereo_params), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Failed to open stereo params: {stereo_params}")

        def read_real(path: str) -> float:
            return float(fs.getNode(path).real())

        def read_int(path: str) -> int:
            return int(fs.getNode(path).real())

        def read_mat(path: str) -> np.ndarray:
            return fs.getNode(path).mat()

        stereo = StereoCamera(
            camera_fx=read_real("Camera.fx"),
            camera_fy=read_real("Camera.fy"),
            camera_cx=read_real("Camera.cx"),
            camera_cy=read_real("Camera.cy"),
            camera_k1=read_real("Camera.k1"),
            camera_k2=read_real("Camera.k2"),
            camera_p1=read_real("Camera.p1"),
            camera_p2=read_real("Camera.p2"),
            camera_width=read_int("Camera.width"),
            camera_height=read_int("Camera.height"),
            camera_fps=read_real("Camera.fps"),
            camera_bf=read_real("Camera.bf"),
            camera_rgb=read_int("Camera.RGB"),
            th_depth=read_real("ThDepth"),
            left_height=read_int("LEFT.height"),
            left_width=read_int("LEFT.width"),
            left_d=read_mat("LEFT.D"),
            left_k=read_mat("LEFT.K"),
            left_r=read_mat("LEFT.R"),
            left_p=read_mat("LEFT.P"),
            right_height=read_int("RIGHT.height"),
            right_width=read_int("RIGHT.width"),
            right_d=read_mat("RIGHT.D"),
            right_k=read_mat("RIGHT.K"),
            right_r=read_mat("RIGHT.R"),
            right_p=read_mat("RIGHT.P"),
        )

        fs.release()
        return stereo

    def _load_yaml(self, yaml_path: Path) -> dict[str, Any]:
        with yaml_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _load_yaml_if_exists(self, yaml_path: Path) -> dict[str, Any] | None:
        if not yaml_path.exists():
            return None
        return self._load_yaml(yaml_path)

    def process_groundtruth(self) -> GroundtruthData | None:
        if self.gt_info is None:
            return None

        data_csv = self.gt_info.data_csv
        if not data_csv.exists():
            raise FileNotFoundError(f"Groundtruth CSV not found: {data_csv}")

        data = np.loadtxt(
            data_csv,
            delimiter=",",
            comments="#",
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        timestamps = data[:, 0].astype(np.int64)
        timestemp_to_index = {ts: i for i, ts in enumerate(timestamps)}
        p_rs_r = data[:, 1:4]
        q_rs = data[:, 4:8]
        v_rs_r = data[:, 8:11]
        b_w_rs_s = data[:, 11:14]
        b_a_rs_s = data[:, 14:17]

        # define transforms
        T_rs: List[np.ndarray] = []
        for i in range(len(timestamps)):
            R_rs = Rotation(q_rs[i, :], scalar_first=True).as_matrix()
            T_rs_top = np.hstack([R_rs, p_rs_r[[i], :].T])
            T_rs.append(np.vstack([T_rs_top, np.array([0.0, 0.0, 0.0, 1.0])]))

        return GroundtruthData(
            timestamps_to_index=timestemp_to_index,
            timestamps=timestamps,
            p_rs_r=p_rs_r,
            T_rs=np.stack(T_rs),
            v_rs_r=v_rs_r,
            b_w_rs_s=b_w_rs_s,
            b_a_rs_s=b_a_rs_s,
        )

    def get_relative_transform(
        self, timestamp0, timestamp1, camera_frame=False
    ) -> np.ndarray:
        """Get the relative transform between two groundtruth poses by index.
        If camera_frame is True, the transform is returned in the left camera frame using T_BS. Otherwise, it is returned in the robot body frame.
        """
        if self.gt_data is None:
            raise ValueError("Groundtruth data not loaded.")
        # Map from timestamps to indices
        index0 = self.gt_data.timestamps_to_index.get(timestamp0)
        index1 = self.gt_data.timestamps_to_index.get(timestamp1)
        if index0 is None:
            raise KeyError(f"timestamp0 {timestamp0} not found in groundtruth data.")
        if index1 is None:
            raise KeyError(f"timestamp1 {timestamp1} not found in groundtruth data.")

        T_rs = self.gt_data.T_rs
        if index0 < 0 or index0 >= len(T_rs):
            raise IndexError(f"index0 {index0} out of bounds for groundtruth data.")
        if index1 < 0 or index1 >= len(T_rs):
            raise IndexError(f"index1 {index1} out of bounds for groundtruth data.")

        # Compute relative transform from index0 to index1 in the robot frame
        T_b0_b1 = np.linalg.inv(T_rs[index0]) @ T_rs[index1]

        if camera_frame:
            # Transform from robot frame to left camera frame using T_BS
            if self.cam0.sensor.T_bs.size == 0:
                raise ValueError("T_BS not available in cam0 sensor info.")
            T_bs = self.cam0.sensor.T_bs
            T_sb = np.linalg.inv(T_bs)
            T_s0_s1 = T_sb @ T_b0_b1 @ np.linalg.inv(T_sb)
            return T_s0_s1

        return T_s0_s1
    
    def get_stereo_cam_config(self, sigma=0.5) -> StereoCameraConfig:
        """Returns a StereoCameraConfig object based on the loaded stereo camera parameters."""
        if self.stereo_camera is None:
            raise ValueError("Stereo camera parameters not loaded.")

        return StereoCameraConfig(
            cu=self.stereo_camera.camera_cx,
            cv=self.stereo_camera.camera_cy,
            f=self.stereo_camera.camera_fx,
            b=self.stereo_camera.camera_bf / self.stereo_camera.camera_fx,
            sigma=sigma,  # Assuming a default sigma value; adjust as needed
        )

    def plot_groundtruth_trajectory(self, stride: int = 50) -> None:
        if self.gt_data is None:
            raise ValueError("Groundtruth data not loaded.")

        T_rs = self.gt_data.T_rs
        if T_rs.size == 0:
            raise ValueError("Groundtruth poses are empty.")

        positions = T_rs[:, :3, 3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "k-")

        stride = max(1, int(stride))
        axis_length = 0.1
        for i in range(0, T_rs.shape[0], stride):
            R_rs = T_rs[i, :3, :3]
            p_rs = T_rs[i, :3, 3]
            x_axis = p_rs + axis_length * R_rs[:, 0]
            y_axis = p_rs + axis_length * R_rs[:, 1]
            z_axis = p_rs + axis_length * R_rs[:, 2]

            ax.plot(
                [p_rs[0], x_axis[0]], [p_rs[1], x_axis[1]], [p_rs[2], x_axis[2]], "r-"
            )
            ax.plot(
                [p_rs[0], y_axis[0]], [p_rs[1], y_axis[1]], [p_rs[2], y_axis[2]], "g-"
            )
            ax.plot(
                [p_rs[0], z_axis[0]], [p_rs[1], z_axis[1]], [p_rs[2], z_axis[2]], "b-"
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Groundtruth trajectory")
        ax.axis("equal")
        ax.view_init(elev=90, azim=-90)

    def get_image_at_timestamp(
        self, timestamp: int, rectify=True
    ) -> tuple[np.ndarray, np.ndarray]:
        cam0_path = self.cam0.timestamp_to_file.get(timestamp)
        cam1_path = self.cam1.timestamp_to_file.get(timestamp)

        if cam0_path is None:
            raise KeyError(f"Timestamp {timestamp} not found in cam0 mapping.")
        if cam1_path is None:
            raise KeyError(f"Timestamp {timestamp} not found in cam1 mapping.")

        if not cam0_path.exists():
            raise FileNotFoundError(f"cam0 image not found: {cam0_path}")
        if not cam1_path.exists():
            raise FileNotFoundError(f"cam1 image not found: {cam1_path}")

        img0 = cv2.imread(cam0_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(cam1_path, cv2.IMREAD_GRAYSCALE)
        if rectify:
            img0, img1 = self.rectify_image_pair(img0, img1)
        return img0, img1

    def rectify_image_pair(
        self, img0: np.ndarray, img1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.stereo_camera is None:
            raise ValueError("Stereo rectification parameters not loaded.")

        rect_img0 = cv2.remap(
            img0,
            self.cam0_rect_map[0],
            self.cam0_rect_map[1],
            interpolation=cv2.INTER_LINEAR,
        )
        rect_img1 = cv2.remap(
            img1,
            self.cam1_rect_map[0],
            self.cam1_rect_map[1],
            interpolation=cv2.INTER_LINEAR,
        )
        return rect_img0, rect_img1


def disparity_interactive(ds: EurocDataset, index=1000):
    # Retrieve images
    timestamp = list(ds.cam0.timestamp_to_file.keys())[index]
    img_L, img_R = ds.get_image_at_timestamp(timestamp, rectify=True)

    # dummy function
    def nothing(x):
        pass

    # Create a window for sliders
    cv2.namedWindow("SGBM_Tuner", cv2.WINDOW_NORMAL)

    # Create trackbars (sliders)
    cv2.createTrackbar(
        "numDisparities", "SGBM_Tuner", 1, 16, nothing
    )  # Will be multiplied by 16
    cv2.createTrackbar(
        "blockSize", "SGBM_Tuner", 2, 10, nothing
    )  # Will be converted to odd number (2*x + 1)
    cv2.createTrackbar("uniquenessRatio", "SGBM_Tuner", 15, 30, nothing)
    cv2.createTrackbar("speckleWindowSize", "SGBM_Tuner", 100, 200, nothing)
    cv2.createTrackbar("speckleRange", "SGBM_Tuner", 2, 5, nothing)

    while True:
        # Read current slider values
        numDisp = cv2.getTrackbarPos("numDisparities", "SGBM_Tuner") * 16
        bs = cv2.getTrackbarPos("blockSize", "SGBM_Tuner") * 2 + 1
        uniq = cv2.getTrackbarPos("uniquenessRatio", "SGBM_Tuner")
        specWin = cv2.getTrackbarPos("speckleWindowSize", "SGBM_Tuner")
        specRange = cv2.getTrackbarPos("speckleRange", "SGBM_Tuner")

        # Enforce constraints
        if numDisp < 16:
            numDisp = 16
        if bs < 3:
            bs = 3

        # Calculate smooth penalties automatically based on block size
        p1 = 8 * 1 * bs * bs
        p2 = 32 * 1 * bs * bs

        # Setup the matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=numDisp,
            blockSize=bs,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            uniquenessRatio=uniq,
            speckleWindowSize=specWin,
            speckleRange=specRange,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # Compute and normalize
        disp = stereo.compute(img_L, img_R).astype(np.float32) / 16.0
        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        # Show output
        cv2.imshow("Disparity Map", disp_vis)
        cv2.imshow("Left Image", img_L)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def make_disparity_plots(ds: EurocDataset, index):
    timestamp = list(ds.cam0.timestamp_to_file.keys())[index]
    im0_raw, im1_raw = ds.get_image_at_timestamp(timestamp, rectify=False)
    im0_rect, im1_rect = ds.get_image_at_timestamp(timestamp, rectify=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(im0_raw, cmap="gray")
    axes[0, 0].set_title("cam0 (raw)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(im1_raw, cmap="gray")
    axes[0, 1].set_title("cam1 (raw)")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(im0_rect, cmap="gray")
    axes[1, 0].set_title("cam0 (rectified) - disparity overlay")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(im1_rect, cmap="gray")
    axes[1, 1].set_title("cam1 (rectified)")
    axes[1, 1].axis("off")
    fig.tight_layout()

    disparity = get_disparity(im0_rect, im1_rect, plot=False)
    axes[1, 0].imshow(disparity, cmap="jet", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    root = Path("/workspace/experiments/data/Euroc/MH_01_easy")
    ds = EurocDataset(root)
    # ds.plot_groundtruth_trajectory()

    # Disparity Tuning:
    # disparity_interactive(ds, 2300)

    # Disparity Check:
    make_disparity_plots(ds, 2000)

    print("done")
