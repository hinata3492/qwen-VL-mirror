# coding: utf-8

import os

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from config import (
    MODEL_NAME,
    PROMPT,
    training_root,
    CHECKPOINT_DIR,
)
from dataset import MVMDPairDataset
from model import QwenMirrorSegmentation


MAX_SAMPLES = 10

OUTPUT_DIR = "./outputs/debug_overfit"


def prepare_inputs(
    processor,
    target_path,
    reference_path,
    prompt,
    device,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": target_path,
                },
                {
                    "type": "image",
                    "image": reference_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: (
            value.to(device)
            if hasattr(value, "to")
            else value
        )
        for key, value in inputs.items()
    }

    return inputs


def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


def save_mask(mask, path):
    """
    mask:
        H x W
        0 or 1
    """

    mask = (
        mask.astype(np.uint8) * 255
    )

    Image.fromarray(mask).save(
        path
    )


def make_comparison(
    target_path,
    gt,
    pred,
    save_path,
):
    """
    Target RGB | GT | Prediction
    を横並びで保存
    """

    target = Image.open(
        target_path
    ).convert("RGB")

    width, height = target.size

    gt_img = Image.fromarray(
        (
            gt.astype(np.uint8)
            * 255
        )
    ).convert("RGB")

    pred_img = Image.fromarray(
        (
            pred.astype(np.uint8)
            * 255
        )
    ).convert("RGB")

    gt_img = gt_img.resize(
        (width, height),
        Image.NEAREST,
    )

    pred_img = pred_img.resize(
        (width, height),
        Image.NEAREST,
    )

    canvas = Image.new(
        "RGB",
        (
            width * 3,
            height,
        ),
    )

    canvas.paste(
        target,
        (0, 0)
    )

    canvas.paste(
        gt_img,
        (width, 0)
    )

    canvas.paste(
        pred_img,
        (width * 2, 0)
    )

    canvas.save(
        save_path
    )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================
    # Dataset
    # ========================================

    print("Loading dataset...")

    dataset = MVMDPairDataset(
        training_root
    )

    num_samples = min(
        MAX_SAMPLES,
        len(dataset)
    )

    # ========================================
    # Processor
    # ========================================

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    # ========================================
    # Model
    # ========================================

    print("Loading model...")

    model = QwenMirrorSegmentation()

    device = next(
        model.qwen.parameters()
    ).device

    model.seg_head = model.seg_head.to(
        device
    )

    # ========================================
    # Load checkpoint
    # ========================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "mirror_head_debug_best.pt"
    )

    print(
        "Loading checkpoint:",
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.seg_head.load_state_dict(
        checkpoint["seg_head"]
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    print(
        "Checkpoint loss:",
        checkpoint.get(
            "loss",
            "unknown"
        )
    )

    model.qwen.eval()
    model.seg_head.eval()

    # ========================================
    # Inference
    # ========================================

    ious = []

    with torch.no_grad():

        for index in range(
            num_samples
        ):

            sample = dataset[index]

            mask = sample["mask"]

            # [1, H, W]
            gt = (
                mask[0]
                .cpu()
                .numpy()
            )

            output_size = (
                mask.shape[-2],
                mask.shape[-1],
            )

            inputs = prepare_inputs(
                processor=processor,
                target_path=sample[
                    "target_path"
                ],
                reference_path=sample[
                    "reference_path"
                ],
                prompt=PROMPT,
                device=device,
            )

            logits = model(
                inputs,
                target_image_index=0,
                output_size=output_size,
            )

            prob = torch.sigmoid(
                logits
            )

            pred = (
                prob > 0.5
            ).float()

            pred_np = (
                pred[
                    0,
                    0
                ]
                .cpu()
                .numpy()
            )

            iou = compute_iou(
                pred_np,
                gt,
            )

            ious.append(iou)

            # =================================
            # Save
            # =================================

            prefix = (
                f"{index:02d}_"
                f"{sample['scene']}_"
                f"{sample['pair']}"
            )

            pred_path = os.path.join(
                OUTPUT_DIR,
                f"{prefix}_pred.png"
            )

            gt_path = os.path.join(
                OUTPUT_DIR,
                f"{prefix}_gt.png"
            )

            comparison_path = os.path.join(
                OUTPUT_DIR,
                f"{prefix}_comparison.png"
            )

            save_mask(
                pred_np,
                pred_path
            )

            save_mask(
                gt,
                gt_path
            )

            make_comparison(
                target_path=sample[
                    "target_path"
                ],
                gt=gt,
                pred=pred_np,
                save_path=comparison_path,
            )

            print(
                f"[{index + 1:02d}/"
                f"{num_samples:02d}] "
                f"{sample['scene']} "
                f"{sample['pair']} "
                f"| IoU={iou:.4f}"
            )

            del inputs
            del logits
            del prob
            del pred

    # ========================================
    # Summary
    # ========================================

    mean_iou = (
        sum(ious)
        / len(ious)
    )

    print()
    print("==============================")
    print("Debug inference finished")
    print("==============================")

    print(
        f"Mean IoU: {mean_iou:.4f}"
    )

    print(
        f"Saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()