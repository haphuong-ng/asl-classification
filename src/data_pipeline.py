"""
ASL data pipeline extracted from `Understanding & Processing.ipynb`.

Reusable parts kept here:
- project/data path resolution
- dataset indexing
- ImageNumber extraction
- CaptureBlock assignment
- group-aware train/validation split
- class mapping
- processed metadata save/load
- PyTorch transforms
- custom Dataset
- DataLoader creation
- end-to-end sanity checks

Exploratory/audit parts such as visual inspection, MD5 audit, brightness,
SSIM plots, and capture-boundary investigation intentionally remain in
the notebook because they are analysis steps, not runtime model input steps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ============================================================
# 1. Constants — kept consistent with the notebook
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff",
}

IMG_SIZE = 224
BATCH_SIZE = 64

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# 2. Project path resolution
# ============================================================

def find_project_root(start: Optional[Path | str] = None) -> Path:
    """
    Find the repository root robustly.

    Expected repository markers:
      - pyproject.toml
      - data/
      - notebooks/

    This avoids hard-coding paths such as C:\\Ha\\Deep\\...
    and works whether code is launched from:
      - repository root
      - notebooks/
      - src/
      - the parent directory that contains `asl-classification/`
    """
    start_path = Path(start).resolve() if start is not None else Path.cwd().resolve()

    candidates = [start_path, *start_path.parents]

    # Search current directory and its parents.
    for candidate in candidates:
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "data").exists()
        ):
            return candidate

    # Support the user's current layout where VS Code may be opened at
    # C:\\Ha\\Deep while the repository is C:\\Ha\\Deep\\asl-classification.
    child_repo = start_path / "asl-classification"
    if (
        (child_repo / "pyproject.toml").exists()
        and (child_repo / "data").exists()
    ):
        return child_repo

    raise FileNotFoundError(
        "Cannot find the ASL project root. "
        f"Started searching from: {start_path}"
    )


def get_data_paths(
    project_root: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """
    Return the standard data paths used by the project.
    """
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else find_project_root()
    )

    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"

    paths = {
        "project_root": root,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "train_dir": raw_dir / "asl_alphabet_train",
        "test_dir": raw_dir / "asl_alphabet_test",
        "metadata_path": processed_dir / "asl_metadata.csv",
        "class_mapping_path": processed_dir / "class_to_idx.json",
    }

    return paths


def validate_raw_data_paths(paths: Dict[str, Path]) -> None:
    """
    Fail early with a clear error if the raw ASL folders are missing.
    """
    missing = []

    for key in ("train_dir", "test_dir"):
        if not paths[key].exists():
            missing.append(str(paths[key]))

    if missing:
        raise FileNotFoundError(
            "Missing raw dataset folder(s):\n- "
            + "\n- ".join(missing)
            + "\nPlace the Kaggle dataset under data/raw/."
        )


# ============================================================
# 3. Build the image index — notebook Section 2
# ============================================================

def build_train_index(train_dir: Path | str) -> pd.DataFrame:
    """
    Scan `asl_alphabet_train/` and create the central image index.

    Returns columns:
      - Path          : absolute path used internally during preparation
      - Class         : folder name / label
    """
    train_dir = Path(train_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    records = []

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image_path in sorted(class_dir.iterdir()):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                records.append({
                    "Path": str(image_path.resolve()),
                    "Class": class_name,
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            f"No supported images were found in: {train_dir}"
        )

    return df


# ============================================================
# 4. Sequence metadata — notebook Sections 7 and 10
# ============================================================

def extract_image_number(path: Path | str) -> Optional[int]:
    """
    Extract the numeric suffix from filenames.

    Example:
        A1001.jpg -> 1001
    """
    filename = Path(path).stem
    match = re.search(r"(\d+)$", filename)

    return int(match.group(1)) if match else None


def assign_capture_block(image_number: int) -> Optional[int]:
    """
    Final capture-block definition from the notebook.

      Block 1:    1–1000
      Block 2: 1001–2000
      Block 3: 2001–2500
      Block 4: 2501–3000
    """
    if 1 <= image_number <= 1000:
        return 1
    if 1001 <= image_number <= 2000:
        return 2
    if 2001 <= image_number <= 2500:
        return 3
    if 2501 <= image_number <= 3000:
        return 4
    return None


def add_capture_metadata(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Add ImageNumber and CaptureBlock to the image index.
    """
    df = df_train.copy()

    df["ImageNumber"] = df["Path"].apply(extract_image_number)
    df["CaptureBlock"] = df["ImageNumber"].apply(assign_capture_block)

    if df["ImageNumber"].isna().any():
        bad = int(df["ImageNumber"].isna().sum())
        raise ValueError(
            f"{bad} image(s) do not have a valid numeric filename suffix."
        )

    if df["CaptureBlock"].isna().any():
        bad = int(df["CaptureBlock"].isna().sum())
        raise ValueError(
            f"{bad} image(s) fall outside the expected capture-block ranges."
        )

    df["ImageNumber"] = df["ImageNumber"].astype(int)
    df["CaptureBlock"] = df["CaptureBlock"].astype(int)

    return df


# ============================================================
# 5. Group-aware split — notebook Section 11
# ============================================================

def split_train_val(
    df_train: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep Blocks 1–3 for training and Block 4 for validation.

    This exactly follows the notebook's leakage-control decision.
    """
    required = {"Class", "CaptureBlock", "Path"}
    missing = required - set(df_train.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    train_df = (
        df_train[
            df_train["CaptureBlock"].isin([1, 2, 3])
        ]
        .reset_index(drop=True)
    )

    val_df = (
        df_train[
            df_train["CaptureBlock"] == 4
        ]
        .reset_index(drop=True)
    )

    overlap = set(train_df["Path"]) & set(val_df["Path"])
    if overlap:
        raise RuntimeError(
            f"Train/validation leakage detected: {len(overlap)} overlapping path(s)."
        )

    return train_df, val_df


def build_class_mapping(
    df_train: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build the same alphabetical class mapping used in the notebook.
    """
    class_names = sorted(df_train["Class"].unique())

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(class_names)
    }

    idx_to_class = {
        idx: class_name
        for class_name, idx in class_to_idx.items()
    }

    return class_to_idx, idx_to_class


# ============================================================
# 6. Processed artifacts
# ============================================================

def build_metadata_table(
    df_train: pd.DataFrame,
    project_root: Path | str,
) -> pd.DataFrame:
    """
    Convert the prepared DataFrame into a portable manifest.

    Important:
    `RelativePath` is saved instead of an absolute Windows path so that
    the same metadata can be used on another teammate's computer.
    """
    root = Path(project_root).resolve()
    metadata = df_train.copy()

    metadata["Split"] = metadata["CaptureBlock"].map({
        1: "train",
        2: "train",
        3: "train",
        4: "val",
    })

    def to_relative(path_value: str) -> str:
        path = Path(path_value).resolve()
        try:
            return path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Image path is outside the project root: {path}"
            ) from exc

    metadata["RelativePath"] = metadata["Path"].apply(to_relative)

    metadata = metadata[
        [
            "RelativePath",
            "Class",
            "ImageNumber",
            "CaptureBlock",
            "Split",
        ]
    ].copy()

    return metadata


def save_processed_artifacts(
    metadata: pd.DataFrame,
    class_to_idx: Dict[str, int],
    processed_dir: Path | str,
) -> Tuple[Path, Path]:
    """
    Save the two processed artifacts needed directly by model notebooks:
      - asl_metadata.csv
      - class_to_idx.json
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = processed_dir / "asl_metadata.csv"
    class_mapping_path = processed_dir / "class_to_idx.json"

    metadata.to_csv(metadata_path, index=False)

    with class_mapping_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            class_to_idx,
            f,
            indent=4,
            ensure_ascii=False,
        )

    return metadata_path, class_mapping_path


def load_processed_artifacts(
    project_root: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Load metadata and class mapping previously generated by this pipeline.
    """
    paths = get_data_paths(project_root)

    metadata_path = paths["metadata_path"]
    class_mapping_path = paths["class_mapping_path"]

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Processed metadata not found: {metadata_path}"
        )

    if not class_mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {class_mapping_path}"
        )

    metadata = pd.read_csv(metadata_path)

    with class_mapping_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        class_to_idx = json.load(f)

    class_to_idx = {
        str(k): int(v)
        for k, v in class_to_idx.items()
    }

    idx_to_class = {
        idx: class_name
        for class_name, idx in class_to_idx.items()
    }

    return metadata, class_to_idx, idx_to_class


def prepare_processed_metadata(
    project_root: Optional[Path | str] = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Reproduce the notebook's fixed preparation logic without rerunning
    exploratory MD5/SSIM/brightness analysis.

    Pipeline:
        raw train folders
        -> image index
        -> ImageNumber
        -> CaptureBlock
        -> Split
        -> portable metadata
        -> class mapping
    """
    paths = get_data_paths(project_root)
    validate_raw_data_paths(paths)

    df_train = build_train_index(paths["train_dir"])
    df_train = add_capture_metadata(df_train)

    class_to_idx, idx_to_class = build_class_mapping(df_train)

    metadata = build_metadata_table(
        df_train=df_train,
        project_root=paths["project_root"],
    )

    if save:
        save_processed_artifacts(
            metadata=metadata,
            class_to_idx=class_to_idx,
            processed_dir=paths["processed_dir"],
        )

    return metadata, class_to_idx, idx_to_class


def load_or_prepare_metadata(
    project_root: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Prefer existing processed artifacts.
    If they do not exist locally, regenerate them from raw data.

    This matches the team's Git workflow:
      - raw and processed data stay local
      - notebook / code is pushed
      - each teammate can regenerate processed metadata locally
    """
    paths = get_data_paths(project_root)

    if (
        paths["metadata_path"].exists()
        and paths["class_mapping_path"].exists()
    ):
        return load_processed_artifacts(paths["project_root"])

    return prepare_processed_metadata(
        project_root=paths["project_root"],
        save=True,
    )


# ============================================================
# 7. Group-aware split from processed metadata
# ============================================================

def split_metadata(
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the portable metadata table using the saved Split column.
    """
    required = {
        "RelativePath",
        "Class",
        "ImageNumber",
        "CaptureBlock",
        "Split",
    }

    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    train_df = (
        metadata[
            metadata["Split"] == "train"
        ]
        .reset_index(drop=True)
    )

    val_df = (
        metadata[
            metadata["Split"] == "val"
        ]
        .reset_index(drop=True)
    )

    return train_df, val_df


# ============================================================
# 8. Preprocessing / augmentation — notebook Section 13
# ============================================================

def get_train_transform(
    img_size: int = IMG_SIZE,
) -> transforms.Compose:
    """
    Train transform copied from the notebook.

    No horizontal flip is used because mirroring a hand sign may change
    the semantic/physical validity of ASL samples.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),

        transforms.RandomRotation(10),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_val_transform(
    img_size: int = IMG_SIZE,
) -> transforms.Compose:
    """
    Deterministic validation transform copied from the notebook.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


# ============================================================
# 9. Custom Dataset — notebook Section 14
# ============================================================

class ASLDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset.

    Each __getitem__ call:
      1. reads one metadata row
      2. opens one raw image
      3. converts class string -> integer label
      4. applies the requested transform
      5. returns (image_tensor, label)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        project_root: Path | str,
        class_to_idx: Dict[str, int],
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.project_root = Path(project_root).resolve()
        self.class_to_idx = class_to_idx
        self.transform = transform

        required = {"RelativePath", "Class"}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Dataset DataFrame is missing columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = (
            self.project_root
            / Path(row["RelativePath"])
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        class_name = row["Class"]

        if class_name not in self.class_to_idx:
            raise KeyError(
                f"Unknown class '{class_name}' for image: {image_path}"
            )

        label = self.class_to_idx[class_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# 10. Dataset + DataLoader creation — notebook Section 15
# ============================================================

def create_datasets(
    metadata: pd.DataFrame,
    project_root: Path | str,
    class_to_idx: Dict[str, int],
    img_size: int = IMG_SIZE,
) -> Tuple[ASLDataset, ASLDataset]:
    """
    Create train and validation datasets with the notebook's transforms.
    """
    train_df, val_df = split_metadata(metadata)

    train_dataset = ASLDataset(
        dataframe=train_df,
        project_root=project_root,
        class_to_idx=class_to_idx,
        transform=get_train_transform(img_size),
    )

    val_dataset = ASLDataset(
        dataframe=val_df,
        project_root=project_root,
        class_to_idx=class_to_idx,
        transform=get_val_transform(img_size),
    )

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders using the stable Windows/Jupyter settings that
    succeeded in the notebook.

    Defaults:
      - batch_size = 64
      - train shuffle = True
      - validation shuffle = False
      - num_workers = 0
      - pin_memory = False
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


# ============================================================
# 11. End-to-end helper for model notebooks
# ============================================================

def build_model_input_pipeline(
    project_root: Optional[Path | str] = None,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    One-call entry point for Simple CNN / Complex CNN / Transfer Learning.

    Returns:
        train_loader,
        val_loader,
        class_to_idx,
        idx_to_class,
        metadata

    Behavior:
      - loads processed metadata if it exists
      - otherwise regenerates it from raw data
      - builds Dataset objects
      - builds DataLoaders
    """
    paths = get_data_paths(project_root)

    metadata, class_to_idx, idx_to_class = load_or_prepare_metadata(
        paths["project_root"]
    )

    train_dataset, val_dataset = create_datasets(
        metadata=metadata,
        project_root=paths["project_root"],
        class_to_idx=class_to_idx,
        img_size=img_size,
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        class_to_idx,
        idx_to_class,
        metadata,
    )


# ============================================================
# 12. Sanity checks
# ============================================================

def validate_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_to_idx: Dict[str, int],
    metadata: pd.DataFrame,
) -> None:
    """
    Run lightweight assertions before training a model.
    """
    if len(class_to_idx) != 29:
        raise AssertionError(
            f"Expected 29 classes, found {len(class_to_idx)}."
        )

    train_count = int((metadata["Split"] == "train").sum())
    val_count = int((metadata["Split"] == "val").sum())

    if train_count != 72_500:
        raise AssertionError(
            f"Expected 72,500 train images, found {train_count}."
        )

    if val_count != 14_500:
        raise AssertionError(
            f"Expected 14,500 validation images, found {val_count}."
        )

    images, labels = next(iter(train_loader))

    if images.ndim != 4:
        raise AssertionError(
            f"Expected image batch with 4 dimensions, got shape {tuple(images.shape)}."
        )

    if images.shape[1:] != (3, IMG_SIZE, IMG_SIZE):
        raise AssertionError(
            "Unexpected image shape: "
            f"{tuple(images.shape)}; expected [B, 3, {IMG_SIZE}, {IMG_SIZE}]."
        )

    if labels.ndim != 1:
        raise AssertionError(
            f"Expected label vector [B], got shape {tuple(labels.shape)}."
        )

    if labels.dtype != torch.int64:
        raise AssertionError(
            f"Expected torch.int64 labels, got {labels.dtype}."
        )

    if labels.min().item() < 0 or labels.max().item() >= len(class_to_idx):
        raise AssertionError(
            "Label index is outside the valid class range."
        )

    print("Pipeline validation passed.")
    print("Train images:", train_count)
    print("Validation images:", val_count)
    print("Classes:", len(class_to_idx))
    print("Train batch shape:", tuple(images.shape))
    print("Label batch shape:", tuple(labels.shape))
    print(
        "Batch min/max:",
        images.min().item(),
        images.max().item(),
    )


# ============================================================
# 13. Example CLI execution
# ============================================================

if __name__ == "__main__":
    root = find_project_root()

    (
        train_loader,
        val_loader,
        class_to_idx,
        idx_to_class,
        metadata,
    ) = build_model_input_pipeline(
        project_root=root,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    validate_pipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_idx=class_to_idx,
        metadata=metadata,
    )
