# coding: utf-8

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import training_root


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
]


def find_file(root_dir, file_id):
    """
    root_dir 内から

        0001.jpg
        0001.png
        0001.jpeg
        ...

    のように、拡張子を変えながら探索する。
    """

    root_dir = Path(root_dir)

    for ext in IMAGE_EXTENSIONS:
        path = root_dir / f"{file_id}{ext}"

        if path.exists():
            return path

    return None


class MVMDPairDataset(Dataset):
    """
    MVMDのpair構造から、

        target image
        reference image
        targetに対応するGT mask

    を返すDataset。

    例:
        pair_0001_0015

        target    = 0001
        reference = 0015
        GT        = SegmentationClassPNG/0001.*
    """

    def __init__(self, root_dir):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.samples = []

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root_dir}"
            )

        self._build_sample_list()

        print()
        print(f"Loaded {len(self.samples)} pairs from:")
        print(self.root_dir)

    def _build_sample_list(self):

        scene_dirs = sorted([
            path
            for path in self.root_dir.iterdir()
            if path.is_dir()
        ])

        for scene_dir in scene_dirs:

            pair_root = (
                scene_dir
                / "pair"
            )

            mask_root = (
                scene_dir
                / "SegmentationClassPNG"
            )

            # -------------------------
            # pair directory check
            # -------------------------

            if not pair_root.exists():
                print(
                    f"[Skip scene] pair directory not found: "
                    f"{scene_dir}"
                )
                continue

            # -------------------------
            # mask directory check
            # -------------------------

            if not mask_root.exists():
                print(
                    f"[Skip scene] SegmentationClassPNG "
                    f"not found: {scene_dir}"
                )
                continue

            pair_dirs = sorted([
                path
                for path in pair_root.iterdir()
                if path.is_dir()
            ])

            for pair_dir in pair_dirs:

                pair_name = pair_dir.name

                # 例:
                # pair_0001_0015

                if not pair_name.startswith("pair_"):
                    continue

                parts = pair_name.split("_")

                if len(parts) != 3:
                    print(
                        f"[Skip] Unexpected pair name: "
                        f"{pair_name}"
                    )
                    continue

                target_id = parts[1]
                reference_id = parts[2]

                image_root = (
                    pair_dir
                    / "JPEGImages_pair"
                )

                if not image_root.exists():
                    print(
                        f"[Skip] JPEGImages_pair missing: "
                        f"{image_root}"
                    )
                    continue

                # -------------------------
                # target
                # -------------------------

                target_path = find_file(
                    image_root,
                    target_id
                )

                if target_path is None:
                    print(
                        f"[Skip] Target image missing: "
                        f"{image_root}/{target_id}.*"
                    )
                    continue

                # -------------------------
                # reference
                # -------------------------

                reference_path = find_file(
                    image_root,
                    reference_id
                )

                if reference_path is None:
                    print(
                        f"[Skip] Reference image missing: "
                        f"{image_root}/{reference_id}.*"
                    )
                    continue

                # -------------------------
                # GT mask
                # -------------------------

                mask_path = find_file(
                    mask_root,
                    target_id
                )

                if mask_path is None:
                    print(
                        f"[Skip] GT mask missing: "
                        f"{mask_root}/{target_id}.*"
                    )
                    continue

                # -------------------------
                # valid sample
                # -------------------------

                self.samples.append({
                    "scene": scene_dir.name,
                    "pair": pair_name,

                    "target_id": target_id,
                    "reference_id": reference_id,

                    "target_path": str(target_path),
                    "reference_path": str(reference_path),
                    "mask_path": str(mask_path),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # ====================================================
        # RGB target image
        # ====================================================

        target = Image.open(
            sample["target_path"]
        ).convert("RGB")

        # ====================================================
        # RGB reference image
        # ====================================================

        reference = Image.open(
            sample["reference_path"]
        ).convert("RGB")

        # ====================================================
        # GT mask
        # ====================================================

        mask = Image.open(
            sample["mask_path"]
        ).convert("L")

        mask = np.asarray(
            mask,
            dtype=np.uint8
        )

        # 0以外をmirror領域として扱う
        mask = (
            mask > 0
        ).astype(np.float32)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        # mask:
        # [1, H, W]

        return {
            "target": target,
            "reference": reference,

            "mask": mask,

            "scene": sample["scene"],
            "pair": sample["pair"],

            "target_id": sample["target_id"],
            "reference_id": sample["reference_id"],

            "target_path": sample["target_path"],
            "reference_path": sample["reference_path"],
            "mask_path": sample["mask_path"],
        }


# ============================================================
# Dataset test
# ============================================================

if __name__ == "__main__":

    dataset = MVMDPairDataset(
        training_root
    )

    print()
    print("===== Dataset =====")
    print(
        "Number of samples:",
        len(dataset)
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "No samples were found."
        )

    sample = dataset[0]

    print()
    print("===== First sample =====")

    print(
        "scene:",
        sample["scene"]
    )

    print(
        "pair:",
        sample["pair"]
    )

    print()

    print(
        "target:",
        sample["target_path"]
    )

    print(
        "reference:",
        sample["reference_path"]
    )

    print(
        "mask:",
        sample["mask_path"]
    )

    print()

    print(
        "target size:",
        sample["target"].size
    )

    print(
        "reference size:",
        sample["reference"].size
    )

    print(
        "mask shape:",
        sample["mask"].shape
    )

    print(
        "mask dtype:",
        sample["mask"].dtype
    )

    print(
        "mask unique:",
        torch.unique(
            sample["mask"]
        )
    )

    print(
        "mirror pixels:",
        int(
            sample["mask"].sum().item()
        )
    )