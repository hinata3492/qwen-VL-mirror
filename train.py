import os
import random
import hashlib
import math

import numpy as np
import torch

from torch.optim import Adam
from tqdm import tqdm

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from config import (
    MODEL_NAME,
    PROMPT,
    training_root,
    validation_root,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
    CHECKPOINT_DIR,
    FEATURE_CACHE_DIR,
)

from dataset import MVMDPairDataset
from model import QwenMirrorSegmentation

# MAGI-MD側と同じLoss
from losses_metrics import BCEDiceLoss


# ============================================================
# Learning-rate schedule
# MAGI-MD側と同様に、最初の3 epochをwarmupし、
# その後はcosine decayで徐々に学習率を下げる
# ============================================================

WARMUP_EPOCHS = 3


# ============================================================
# Root
# config.py が
#   str
# または
#   (path, "scene", "MVMD_train")
# のどちらでも対応
# ============================================================

def resolve_root_arg(root_like):

    if root_like is None:
        return None

    if isinstance(root_like, (tuple, list)):

        if len(root_like) == 0:
            return None

        return os.path.expanduser(
            str(root_like[0])
        )

    return os.path.expanduser(
        str(root_like)
    )


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# IoU
# MAGI-MD側と同じ形式
# ============================================================

def compute_batch_iou(
    logits,
    targets,
    thresh=0.5,
    eps=1e-6,
):

    probs = torch.sigmoid(
        logits
    )

    preds = (
        probs > thresh
    ).float()

    preds = preds.view(
        preds.size(0),
        -1,
    )

    targets = targets.view(
        targets.size(0),
        -1,
    ).float()

    inter = (
        preds * targets
    ).sum(dim=1)

    union = (
        preds.sum(dim=1)
        + targets.sum(dim=1)
        - inter
    )

    iou = (
        inter + eps
    ) / (
        union + eps
    )

    return iou.mean().item()


# ============================================================
# MAE
# ============================================================

def compute_batch_mae(
    logits,
    targets,
):

    probs = torch.sigmoid(
        logits
    )

    return torch.abs(
        probs
        - targets.float()
    ).mean().item()


# ============================================================
# Qwen input
# ============================================================

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

    text = (
        processor
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    (
        image_inputs,
        video_inputs,
    ) = process_vision_info(
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
        for key, value
        in inputs.items()
    }

    return inputs


# ============================================================
# Feature cache path
# ============================================================

def get_cache_path(
    sample,
    split_name,
):

    # 同じ画像ペア + 同じPromptなら
    # Frozen Qwenの特徴は同じ

    key = (
        sample["target_path"]
        + "||"
        + sample["reference_path"]
        + "||"
        + PROMPT
    )

    hash_name = hashlib.md5(
        key.encode("utf-8")
    ).hexdigest()

    cache_dir = os.path.join(
        FEATURE_CACHE_DIR,
        split_name,
    )

    os.makedirs(
        cache_dir,
        exist_ok=True,
    )

    return os.path.join(
        cache_dir,
        f"{hash_name}.pt",
    )


# ============================================================
# Feature取得
#
# cacheあり
#   → .pt を読む
#
# cacheなし
#   → Qwen forward
#   → feature保存
# ============================================================

def get_features(
    model,
    processor,
    sample,
    device,
    split_name,
):

    cache_path = get_cache_path(
        sample,
        split_name,
    )

    # --------------------------------------------------------
    # Cache HIT
    # --------------------------------------------------------

    if os.path.exists(
        cache_path
    ):

        cached = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        visual_feature = (
            cached["visual_feature"]
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        semantic_feature = (
            cached["semantic_feature"]
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        return (
            visual_feature,
            semantic_feature,
            True,
        )

    # --------------------------------------------------------
    # Cache MISS
    # --------------------------------------------------------

    inputs = prepare_inputs(
        processor=processor,

        target_path=
        sample["target_path"],

        reference_path=
        sample["reference_path"],

        prompt=PROMPT,

        device=device,
    )

    (
        visual_feature,
        semantic_feature,
    ) = model.extract_features(
        inputs,
        target_image_index=0,
    )

    # --------------------------------------------------------
    # CPUに一時保存
    # --------------------------------------------------------

    torch.save(
        {
            "visual_feature":
                visual_feature
                .detach()
                .cpu(),

            "semantic_feature":
                semantic_feature
                .detach()
                .cpu(),
        },
        cache_path,
    )

    del inputs

    return (
        visual_feature,
        semantic_feature,
        False,
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    processor,
    dataset,
    device,
    optimizer,
    criterion,
    epoch,
    epochs,
):

    # QwenはFrozen
    model.qwen.eval()

    # Headのみ学習
    model.seg_head.train()

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    indices = list(
        range(len(dataset))
    )

    random.shuffle(
        indices
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_loss = 0.0
    total_iou = 0.0
    total_mae = 0.0

    n = 0

    cache_hits = 0
    cache_misses = 0

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    pbar = tqdm(
        indices,

        desc=
        f"[Train {epoch}/{epochs}]",

        dynamic_ncols=True,
        ascii=True,
        mininterval=1.0,
        smoothing=0.1,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for it, index in enumerate(
        pbar,
        start=1,
    ):

        sample = dataset[index]

        # ====================================================
        # Ground Truth
        # ====================================================

        mask = (
            sample["mask"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        output_size = (
            mask.shape[-2],
            mask.shape[-1],
        )

        # ====================================================
        # Qwen feature
        # ====================================================

        (
            visual_feature,
            semantic_feature,
            from_cache,
        ) = get_features(
            model=model,
            processor=processor,
            sample=sample,
            device=device,
            split_name="train",
        )

        if from_cache:

            cache_hits += 1

        else:

            cache_misses += 1

        # ====================================================
        # Head forward
        # ====================================================

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model.seg_head(
            visual_feature,
            semantic_feature,
            output_size=output_size,
        )

        # ====================================================
        # Safety check
        # ====================================================

        if logits.shape != mask.shape:

            raise RuntimeError(
                "Shape mismatch:\n"
                f"logits={logits.shape}\n"
                f"mask={mask.shape}"
            )

        # ====================================================
        # Loss
        #
        # MAGI-MDと同じ BCEDiceLoss
        # ====================================================

        loss = criterion(
            logits,
            mask,
        )

        # ====================================================
        # Backward
        # ====================================================

        loss.backward()

        optimizer.step()

        # ====================================================
        # Metrics
        # ====================================================

        with torch.no_grad():

            iou = compute_batch_iou(
                logits,
                mask,
                thresh=0.5,
            )

            mae = compute_batch_mae(
                logits,
                mask,
            )

        bs = mask.size(0)

        n += bs

        total_loss += (
            loss.item()
            * bs
        )

        total_iou += (
            iou
            * bs
        )

        total_mae += (
            mae
            * bs
        )

        # ====================================================
        # Display
        # MAGI-MD側に寄せる
        # ====================================================

        if (
            it == 1
            or it % 10 == 0
            or it == len(dataset)
        ):

            pbar.set_postfix(
                loss=
                f"{total_loss / n:.4f}",

                iou=
                f"{total_iou / n:.4f}",

                mae=
                f"{total_mae / n:.4f}",

                cache=
                f"{cache_hits}/{it}",
            )

        # ====================================================
        # Cleanup
        # ====================================================

        del visual_feature
        del semantic_feature
        del logits
        del mask

    return {
        "loss":
            total_loss / n,

        "iou":
            total_iou / n,

        "mae":
            total_mae / n,

        "cache_hits":
            cache_hits,

        "cache_misses":
            cache_misses,
    }


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    processor,
    dataset,
    device,
    criterion,
    epoch,
    epochs,
    iou_thresholds=None,
):

    model.qwen.eval()
    model.seg_head.eval()

    if iou_thresholds is None:

        iou_thresholds = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_loss = 0.0
    total_iou = 0.0
    total_mae = 0.0

    n = 0

    cache_hits = 0
    cache_misses = 0

    # --------------------------------------------------------
    # IoU threshold statistics
    # --------------------------------------------------------

    thresh_stats = {
        th: {
            "sum_iou": 0.0,
            "count": 0,
        }
        for th in iou_thresholds
    }

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    pbar = tqdm(
        range(len(dataset)),

        desc=
        f"[Val   {epoch}/{epochs}]",

        dynamic_ncols=True,
        ascii=True,
        mininterval=1.0,
        smoothing=0.1,
    )

    # --------------------------------------------------------
    # Validation loop
    # --------------------------------------------------------

    with torch.no_grad():

        for it, index in enumerate(
            pbar,
            start=1,
        ):

            sample = dataset[index]

            # =================================================
            # GT
            # =================================================

            mask = (
                sample["mask"]
                .unsqueeze(0)
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            output_size = (
                mask.shape[-2],
                mask.shape[-1],
            )

            # =================================================
            # Feature
            # =================================================

            (
                visual_feature,
                semantic_feature,
                from_cache,
            ) = get_features(
                model=model,
                processor=processor,
                sample=sample,
                device=device,
                split_name="val",
            )

            if from_cache:

                cache_hits += 1

            else:

                cache_misses += 1

            # =================================================
            # Head
            # =================================================

            logits = model.seg_head(
                visual_feature,
                semantic_feature,
                output_size=output_size,
            )

            # =================================================
            # BCEDiceLoss
            # =================================================

            loss = criterion(
                logits,
                mask,
            )

            # =================================================
            # Metrics @ 0.5
            # =================================================

            iou = compute_batch_iou(
                logits,
                mask,
                thresh=0.5,
            )

            mae = compute_batch_mae(
                logits,
                mask,
            )

            bs = mask.size(0)

            n += bs

            total_loss += (
                loss.item()
                * bs
            )

            total_iou += (
                iou
                * bs
            )

            total_mae += (
                mae
                * bs
            )

            # =================================================
            # Threshold sweep
            # =================================================

            probs = torch.sigmoid(
                logits
            )

            mask_f = mask.float()

            for th in iou_thresholds:

                preds = (
                    probs > th
                ).float()

                preds_flat = preds.view(
                    preds.size(0),
                    -1,
                )

                mask_flat = mask_f.view(
                    mask_f.size(0),
                    -1,
                )

                inter = (
                    preds_flat
                    * mask_flat
                ).sum(dim=1)

                union = (
                    preds_flat.sum(dim=1)
                    + mask_flat.sum(dim=1)
                    - inter
                )

                iou_each = (
                    inter + 1e-6
                ) / (
                    union + 1e-6
                )

                thresh_stats[
                    th
                ]["sum_iou"] += (
                    iou_each
                    .sum()
                    .item()
                )

                thresh_stats[
                    th
                ]["count"] += (
                    preds.size(0)
                )

            # =================================================
            # Display
            # =================================================

            if (
                it == 1
                or it % 10 == 0
                or it == len(dataset)
            ):

                pbar.set_postfix(
                    loss=
                    f"{total_loss / n:.4f}",

                    iou=
                    f"{total_iou / n:.4f}",

                    mae=
                    f"{total_mae / n:.4f}",

                    cache=
                    f"{cache_hits}/{it}",
                )

            # =================================================
            # Cleanup
            # =================================================

            del visual_feature
            del semantic_feature
            del logits
            del mask

    # ========================================================
    # Threshold results
    # ========================================================

    thresh_ious = {}

    best_thresh = None
    best_iou = -1.0

    for th in iou_thresholds:

        count = (
            thresh_stats[
                th
            ]["count"]
        )

        if count > 0:

            current_iou = (
                thresh_stats[
                    th
                ]["sum_iou"]
                / count
            )

        else:

            current_iou = 0.0

        thresh_ious[
            th
        ] = current_iou

        if current_iou > best_iou:

            best_iou = current_iou

            best_thresh = th

    return {
        "loss":
            total_loss / n,

        "iou":
            total_iou / n,

        "mae":
            total_mae / n,

        "thresh_ious":
            thresh_ious,

        "best_thresh":
            best_thresh,

        "best_iou":
            best_iou,

        "cache_hits":
            cache_hits,

        "cache_misses":
            cache_misses,
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Seed
    # ========================================================

    set_seed(
        SEED
    )

    # ========================================================
    # Root
    # ========================================================

    train_root = resolve_root_arg(
        training_root
    )

    val_root = resolve_root_arg(
        validation_root
    )

    if train_root is None:

        raise RuntimeError(
            "training_root is None"
        )

    if val_root is None:

        raise RuntimeError(
            "validation_root is None"
        )

    # ========================================================
    # Directories
    # ========================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        FEATURE_CACHE_DIR,
        exist_ok=True,
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "[info] loading training dataset"
    )

    train_dataset = MVMDPairDataset(
        train_root
    )

    print()
    print(
        "[info] loading validation dataset"
    )

    val_dataset = MVMDPairDataset(
        val_root
    )

    print()

    print(
        f"[info] train_root       = "
        f"{train_root}"
    )

    print(
        f"[info] val_root         = "
        f"{val_root}"
    )

    print(
        f"[info] train_samples    = "
        f"{len(train_dataset)}"
    )

    print(
        f"[info] val_samples      = "
        f"{len(val_dataset)}"
    )

    print(
        f"[info] epochs           = "
        f"{NUM_EPOCHS}"
    )

    print(
        f"[info] learning_rate    = "
        f"{LEARNING_RATE}"
    )

    print(
        f"[info] weight_decay     = "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"[info] feature_cache    = "
        f"{FEATURE_CACHE_DIR}"
    )

    # ========================================================
    # Processor
    # ========================================================

    print()
    print(
        "[info] loading processor"
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "[info] loading model"
    )

    model = (
        QwenMirrorSegmentation()
    )

    device = next(
        model.qwen.parameters()
    ).device

    model.seg_head = (
        model.seg_head
        .to(device)
    )

    print(
        f"[info] device           = "
        f"{device}"
    )

    # ========================================================
    # Parameter check
    # ========================================================

    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"[info] trainable params = "
        f"{trainable_count:,}"
    )

    print(
        f"[info] total params     = "
        f"{total_count:,}"
    )

    # Qwenが本当にfreezeされているか確認
    assert all(
        not p.requires_grad
        for p in model.qwen.parameters()
    )

    # ========================================================
    # Loss
    #
    # MAGI-MD/scripts/losses_metrics.py
    # のものをそのまま使用
    # ========================================================

    criterion = BCEDiceLoss()

    print(
        "[info] criterion        = "
        "BCEDiceLoss"
    )

    # ========================================================
    # Optimizer
    # Headだけ更新
    # ========================================================

    param_groups = [
        {
            "params": model.seg_head.parameters(),
            "lr": LEARNING_RATE,
        }
    ]

    optimizer = Adam(
        param_groups,
        betas=(0.9, 0.99),
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # Scheduler
    # MAGI-MD側と同じ warmup + cosine decay
    # ========================================================

    warmup_epochs = max(
        1,
        int(WARMUP_EPOCHS),
    )

    total_epochs = NUM_EPOCHS

    def lr_lambda(current_epoch):

        epoch = current_epoch + 1

        # 最初のWARMUP_EPOCHSで線形warmup
        if epoch <= warmup_epochs:

            return (
                epoch
                / float(warmup_epochs)
            )

        # warmup後はcosine decay
        if total_epochs == warmup_epochs:

            return 1.0

        t = (
            epoch - warmup_epochs
        ) / float(
            total_epochs - warmup_epochs
        )

        return 0.5 * (
            math.cos(
                t * math.pi
            )
            + 1.0
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    print(
        f"[info] warmup_epochs    = "
        f"{WARMUP_EPOCHS}"
    )

    print(
        "[info] scheduler        = "
        "linear warmup + cosine decay"
    )

    # ========================================================
    # IoU thresholds
    # ========================================================

    iou_thresholds = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]

    print(
        f"[info] iou_thresholds   = "
        f"{iou_thresholds}"
    )

    # ========================================================
    # Best
    # ========================================================

    # Qwen版では
    # Val IoU@0.5 最大をbestにする

    best_val_iou_05 = -1.0

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()

        print(
            f"[epoch] "
            f"{epoch}/{NUM_EPOCHS} "
            f"lr_head="
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        # ====================================================
        # Train
        # ====================================================

        train_result = train_one_epoch(
            model=model,
            processor=processor,
            dataset=train_dataset,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            epochs=NUM_EPOCHS,
        )

        # ====================================================
        # Validation
        # ====================================================

        val_result = validate(
            model=model,
            processor=processor,
            dataset=val_dataset,
            device=device,
            criterion=criterion,
            epoch=epoch,
            epochs=NUM_EPOCHS,
            iou_thresholds=iou_thresholds,
        )

        # ====================================================
        # IoU@0.5
        # ====================================================

        val_iou_05 = (
            val_result[
                "thresh_ious"
            ].get(
                0.5,
                val_result["iou"],
            )
        )

        # ====================================================
        # Epoch summary
        # ====================================================

        print(
            f"[epoch {epoch}] "
            f"train_loss="
            f"{train_result['loss']:.4f}, "
            f"train_iou="
            f"{train_result['iou']:.4f}, "
            f"train_mae="
            f"{train_result['mae']:.4f}, "
            f"val_loss="
            f"{val_result['loss']:.4f}, "
            f"val_iou@0.5="
            f"{val_iou_05:.4f}, "
            f"val_mae="
            f"{val_result['mae']:.4f}"
        )

        # ====================================================
        # Threshold summary
        # ====================================================

        show_thresholds = [
            0.3,
            0.5,
            0.7,
        ]

        show_str = ", ".join(
            [
                f"{th:.1f}:"
                f"{val_result['thresh_ious'][th]:.4f}"
                for th in show_thresholds
                if th
                in val_result["thresh_ious"]
            ]
        )

        print(
            f"[epoch {epoch}] "
            f"best_iou="
            f"{val_result['best_iou']:.4f} "
            f"@ th="
            f"{val_result['best_thresh']:.2f}"
            + (
                f" | {show_str}"
                if show_str
                else ""
            )
        )

        # ====================================================
        # Cache information
        # ====================================================

        print(
            f"[cache] "
            f"train hit/miss="
            f"{train_result['cache_hits']}/"
            f"{train_result['cache_misses']}, "
            f"val hit/miss="
            f"{val_result['cache_hits']}/"
            f"{val_result['cache_misses']}"
        )

        # ====================================================
        # Latest checkpoint
        # ====================================================

        latest_path = os.path.join(
            CHECKPOINT_DIR,
            "mirror_head_baseline_latest.pt",
        )

        torch.save(
            {
                "epoch":
                    epoch,

                "seg_head":
                    model.seg_head
                    .state_dict(),

                "optimizer":
                    optimizer
                    .state_dict(),

                "scheduler":
                    scheduler
                    .state_dict(),

                "train_loss":
                    train_result[
                        "loss"
                    ],

                "train_iou":
                    train_result[
                        "iou"
                    ],

                "train_mae":
                    train_result[
                        "mae"
                    ],

                "val_loss":
                    val_result[
                        "loss"
                    ],

                "val_iou_05":
                    val_iou_05,

                "val_mae":
                    val_result[
                        "mae"
                    ],

                "val_best_iou":
                    val_result[
                        "best_iou"
                    ],

                "val_best_thresh":
                    val_result[
                        "best_thresh"
                    ],

                "learning_rate":
                    optimizer
                    .param_groups[0]["lr"],

                "train_root":
                    train_root,

                "val_root":
                    val_root,
            },
            latest_path,
        )

        # ====================================================
        # Best checkpoint
        # ====================================================

        if val_iou_05 > best_val_iou_05:

            best_val_iou_05 = (
                val_iou_05
            )

            best_path = os.path.join(
                CHECKPOINT_DIR,
                "mirror_head_baseline_best.pt",
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "seg_head":
                        model.seg_head
                        .state_dict(),

                    "optimizer":
                        optimizer
                        .state_dict(),

                    "scheduler":
                        scheduler
                        .state_dict(),

                    "train_loss":
                        train_result[
                            "loss"
                        ],

                    "train_iou":
                        train_result[
                            "iou"
                        ],

                    "train_mae":
                        train_result[
                            "mae"
                        ],

                    "val_loss":
                        val_result[
                            "loss"
                        ],

                    "val_iou_05":
                        val_iou_05,

                    "val_mae":
                        val_result[
                            "mae"
                        ],

                    "val_best_iou":
                        val_result[
                            "best_iou"
                        ],

                    "val_best_thresh":
                        val_result[
                            "best_thresh"
                        ],

                    "learning_rate":
                        optimizer
                        .param_groups[0]["lr"],

                    "train_root":
                        train_root,

                    "val_root":
                        val_root,
                },
                best_path,
            )

            print(
                f"[best] "
                f"epoch={epoch}, "
                f"val_iou@0.5="
                f"{val_iou_05:.4f}, "
                f"saved="
                f"{best_path}"
            )

        # ====================================================
        # Scheduler step
        # MAGI-MD側と同様にepoch末で更新
        # ====================================================

        scheduler.step()

    # ========================================================
    # Finish
    # ========================================================

    print()

    print(
        f"[done] "
        f"best_val_iou@0.5="
        f"{best_val_iou_05:.4f}"
    )

    print(
        f"[done] "
        f"feature cache remains at "
        f"{FEATURE_CACHE_DIR}"
    )

    print(
        "[done] "
        "学習結果を確認した後に削除できます:"
    )

    print(
        f"rm -rf {FEATURE_CACHE_DIR}"
    )


if __name__ == "__main__":
    main()