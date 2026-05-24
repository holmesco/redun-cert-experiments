from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraInfo:
    path: Path
    sensor: dict[str, Any]
    data_dir: Path
    data_csv: Path
    timestamp_to_file: dict[int, Path]


@dataclass(frozen=True)
class GroundtruthInfo:
    path: Path
    sensor: dict[str, Any] | None
    data_csv: Path


@dataclass(frozen=True)
class GroundtruthData:
    timestamps: np.ndarray
    p_rs_r: np.ndarray
    q_rs: np.ndarray
    v_rs_r: np.ndarray
    b_w_rs_s: np.ndarray
    b_a_rs_s: np.ndarray


class EurocDataset:
    def __init__(self, path: Path):
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
        self.groundtruth = self._load_groundtruth(
            mav0_dir / "state_groundtruth_estimate0"
        )
        self.gt_data = self.process_groundtruth()

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

        return CameraInfo(
            path=cam_dir,
            sensor=self._load_yaml(sensor_yaml),
            data_dir=cam_dir / "data",
            data_csv=data_csv,
            timestamp_to_file=self._load_camera_mapping(data_csv, cam_dir / "data"),
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

    def _load_groundtruth(self, gt_dir: Path) -> GroundtruthInfo | None:
        if not gt_dir.exists():
            return None

        sensor_yaml = gt_dir / "sensor.yaml"
        return GroundtruthInfo(
            path=gt_dir,
            sensor=self._load_yaml_if_exists(sensor_yaml),
            data_csv=gt_dir / "data.csv",
        )

    def process_groundtruth(self) -> GroundtruthData | None:
        if self.groundtruth is None:
            return None

        data_csv = self.groundtruth.data_csv
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
        p_rs_r = data[:, 1:4]
        q_rs = data[:, 4:8]
        v_rs_r = data[:, 8:11]
        b_w_rs_s = data[:, 11:14]
        b_a_rs_s = data[:, 14:17]

        return GroundtruthData(
            timestamps=timestamps,
            p_rs_r=p_rs_r,
            q_rs=q_rs,
            v_rs_r=v_rs_r,
            b_w_rs_s=b_w_rs_s,
            b_a_rs_s=b_a_rs_s,
        )

    def _load_yaml(self, yaml_path: Path) -> dict[str, Any]:
        with yaml_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _load_yaml_if_exists(self, yaml_path: Path) -> dict[str, Any] | None:
        if not yaml_path.exists():
            return None
        return self._load_yaml(yaml_path)


if __name__ == "__main__":
    root = Path("/workspace/experiments/data/Euroc/V1_01_easy")
    ds = EurocDataset(root)

    print("done")
