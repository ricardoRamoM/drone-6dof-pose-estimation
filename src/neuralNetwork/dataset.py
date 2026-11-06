# dataset.py
# Dataset de imágenes RGB etiquetadas con poses de COLMAP.
# Responsabilidades:
#   - Leer el CSV generado por src/post_colmap/split_dataset.py
#   - Filtrar por split (train / val / test)
#   - Verificar que las imágenes existan en disco
#   - Aplicar transformaciones (resize + normalización ImageNet)
#   - Devolver (imagen, pose, image_name) por índice
#
# Estructura esperada:
#   dataset/<version>/frames_all/<video_stem>_frame_00000.jpg
#   dataset/<version>/poses_split.csv

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_DATASET_VERSION = "v2_0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"

POSE_COLUMNS = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def poses_csv_path(version=DEFAULT_DATASET_VERSION) -> Path:
    return DATASET_ROOT / version / "poses_split.csv"


def image_dir_path(version=DEFAULT_DATASET_VERSION) -> Path:
    return DATASET_ROOT / version / "frames_all"


class ColmapPoseDataset(Dataset):
    """Dataset de imágenes RGB etiquetadas con poses de COLMAP."""

    def __init__(
        self,
        split="train",
        image_size=224,
        version=DEFAULT_DATASET_VERSION,
        csv_path=None,
        image_dir=None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Split inválido: {split}. Usa train, val o test."
            )

        if csv_path is None:
            csv_path = poses_csv_path(version)
        if image_dir is None:
            image_dir = image_dir_path(version)

        self.version = version
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)

        self.df = pd.read_csv(self.csv_path)

        required_columns = {"image_name", *POSE_COLUMNS, "split"}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Faltan columnas en el CSV: {sorted(missing_columns)}"
            )

        if self.df["image_name"].duplicated().any():
            raise ValueError(
                "El CSV contiene nombres de imagen duplicados."
            )

        self.df = self.df[self.df["split"] == split].copy()
        self.df = self.df.reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"No hay imágenes para el conjunto {split}.")

        self.df[POSE_COLUMNS] = self.df[POSE_COLUMNS].apply(
            pd.to_numeric, errors="raise"
        )

        if not self.df[POSE_COLUMNS].notna().all().all():
            raise ValueError("Hay etiquetas de pose con valores NaN.")

        if not self.df[POSE_COLUMNS].map(
            lambda value: abs(value) != float("inf")
        ).all().all():
            raise ValueError("Hay etiquetas de pose infinitas.")

        missing_images = [
            name
            for name in self.df["image_name"]
            if not (self.image_dir / name).is_file()
        ]

        if missing_images:
            raise FileNotFoundError(
                f"Faltan {len(missing_images)} imágenes en {self.image_dir}. "
                f"Ejemplos: {missing_images[:5]}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = self.image_dir / row["image_name"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        pose = torch.tensor(
            row[POSE_COLUMNS].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

        return image, pose, row["image_name"]