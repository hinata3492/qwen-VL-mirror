# coding: utf-8

import os
import hashlib

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from config import (
    MODEL_NAME,
    PROMPT,
    test_root,
    CHECKPOINT_DIR,
    FEATURE_CACHE_DIR,
)

from dataset import MVMDPairDataset
from model import QwenMirrorSegmentation


# ============================================================
# Output
# ============================================================

EVAL_OUTPUT_DIR = "./eval_outputs"


# ============================================================
# Root
# str / tuple の両方に対応
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
# Metrics
# MAGI-MD側の評価コードに合わせる
# ============================================================

class Metrics:

    def __init__(self, thr=0.5):

        self.thr = float(thr)
        self.initial()


    def initial(self):

        self.tp = []
        self.tn = []
        self.fp = []
        self.fn = []

        self.precision = []
        self.recall = []

        self.cnt = 0

        self.mae = []
        self.tot = []

        self.image_iou = []


    # ========================================================
    # 1サンプル追加
    # ========================================================

    def update(
        self,
        pred,
        target,
        name="",
    ):

        pred = pred.reshape(-1)
        target = target.reshape(-1)

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        assert (
            pred.min() >= 0.0
            and pred.max() <= 1.0
        ), (
            f"pred out of range: {name}"
        )

        assert (
            target.min() >= 0.0
            and target.max() <= 1.0
        ), (
            f"target out of range: {name}"
        )

        assert (
            pred.shape == target.shape
        ), (
            f"shape mismatch: {name}, "
            f"pred={pred.shape}, "
            f"target={target.shape}"
        )

        # ----------------------------------------------------
        # Binary
        # MAGI-MD側と同じ >=
        # ----------------------------------------------------

        pred_bin = (
            pred >= self.thr
        )

        true_bin = (
            target > 0.5
        )

        # ----------------------------------------------------
        # TP / TN / FP / FN
        # ----------------------------------------------------

        tp = int(
            np.logical_and(
                pred_bin,
                true_bin,
            ).sum()
        )

        tn = int(
            np.logical_and(
                ~pred_bin,
                ~true_bin,
            ).sum()
        )

        fp = int(
            np.logical_and(
                pred_bin,
                ~true_bin,
            ).sum()
        )

        fn = int(
            np.logical_and(
                ~pred_bin,
                true_bin,
            ).sum()
        )

        self.tp.append(tp)
        self.tn.append(tn)
        self.fp.append(fp)
        self.fn.append(fn)

        self.tot.append(
            target.shape[0]
        )

        # ----------------------------------------------------
        # Foreground IoU
        # ----------------------------------------------------

        union_fg = (
            tp
            + fp
            + fn
        )

        if union_fg == 0:

            iou_fg = 1.0

        else:

            iou_fg = (
                tp / union_fg
            )

        self.image_iou.append(
            iou_fg
        )

        # ----------------------------------------------------
        # F-beta用
        # ----------------------------------------------------

        eps = 1e-9

        pred_255 = np.clip(
            np.round(
                pred * 255.0
            ),
            0,
            255,
        ).astype(
            np.uint8
        )

        bins = np.arange(
            257
        )

        fg_hist, _ = np.histogram(
            pred_255[
                true_bin
            ],
            bins=bins,
        )

        bg_hist, _ = np.histogram(
            pred_255[
                ~true_bin
            ],
            bins=bins,
        )

        fg_w_thrs = np.cumsum(
            np.flip(
                fg_hist
            ),
            axis=0,
        )

        bg_w_thrs = np.cumsum(
            np.flip(
                bg_hist
            ),
            axis=0,
        )

        TPs = (
            fg_w_thrs
            .astype(
                np.float64
            )
        )

        Ps = (
            fg_w_thrs
            + bg_w_thrs
        ).astype(
            np.float64
        )

        T = max(
            int(
                np.count_nonzero(
                    true_bin
                )
            ),
            1,
        )

        precisions = (
            TPs + eps
        ) / (
            Ps + eps
        )

        recalls = (
            TPs + eps
        ) / (
            T + eps
        )

        self.precision.append(
            precisions
        )

        self.recall.append(
            recalls
        )

        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        self.mae.append(
            float(
                np.mean(
                    np.abs(
                        pred
                        - target
                    )
                )
            )
        )

        self.cnt += 1


    # ========================================================
    # Foreground IoU
    # ========================================================

    def compute_fg_iou(self):

        vals = []

        for i in range(
            len(self.tp)
        ):

            union = (
                self.tp[i]
                + self.fp[i]
                + self.fn[i]
            )

            if union == 0:

                vals.append(
                    1.0
                )

            else:

                vals.append(
                    self.tp[i]
                    / union
                )

        if len(vals) == 0:
            return 0.0

        return float(
            np.mean(vals)
        )


    # ========================================================
    # Background IoU
    # ========================================================

    def compute_bg_iou(self):

        vals = []

        for i in range(
            len(self.tn)
        ):

            union = (
                self.tn[i]
                + self.fp[i]
                + self.fn[i]
            )

            if union == 0:

                vals.append(
                    1.0
                )

            else:

                vals.append(
                    self.tn[i]
                    / union
                )

        if len(vals) == 0:
            return 0.0

        return float(
            np.mean(vals)
        )


    # ========================================================
    # mIoU
    # ========================================================

    def compute_miou_2class(self):

        return (
            self.compute_fg_iou()
            + self.compute_bg_iou()
        ) / 2.0


    # ========================================================
    # Per-image IoU
    # ========================================================

    def per_image_iou_list(self):

        return np.array(
            self.image_iou,
            dtype=np.float64,
        )


    # ========================================================
    # IoU std
    # ========================================================

    def compute_iou_std(self):

        iou = (
            self.per_image_iou_list()
        )

        if len(iou) == 0:
            return 0.0

        return float(
            np.std(iou)
        )


    # ========================================================
    # F-beta
    # ========================================================

    def compute_fbeta(
        self,
        beta=0.3,
    ):

        if (
            len(self.precision) == 0
            or len(self.recall) == 0
        ):

            return 0.0

        beta_square = (
            beta ** 2
        )

        precision = np.array(
            self.precision
        ).mean(
            axis=0
        )

        recall = np.array(
            self.recall
        ).mean(
            axis=0
        )

        fvals = (
            (1 + beta_square)
            * precision
            * recall
        ) / (
            beta_square
            * precision
            + recall
            + 1e-12
        )

        return float(
            np.max(fvals)
        )


    # ========================================================
    # MAE
    # ========================================================

    def compute_mae(self):

        if len(self.mae) == 0:
            return 0.0

        return float(
            np.mean(
                self.mae
            )
        )


    # ========================================================
    # Accuracy
    # ========================================================

    def accuracy(self):

        if len(self.tot) == 0:
            return 0.0

        vals = [
            (
                self.tp[i]
                + self.tn[i]
            )
            / self.tot[i]

            for i in range(
                len(self.tot)
            )
        ]

        return float(
            np.mean(vals)
        )


    # ========================================================
    # BER
    # ========================================================

    def ber(self):

        if len(self.tot) == 0:
            return 0.0

        vals = []

        for i in range(
            len(self.tot)
        ):

            pos = (
                self.tp[i]
                + self.fn[i]
            )

            neg = (
                self.tn[i]
                + self.fp[i]
            )

            if (
                pos == 0
                and neg == 0
            ):

                continue

            elif pos == 0:

                vals.append(
                    100.0
                    * (
                        self.fp[i]
                        / max(
                            neg,
                            1,
                        )
                    )
                )

            elif neg == 0:

                vals.append(
                    100.0
                    * (
                        1.0
                        - self.tp[i]
                        / max(
                            pos,
                            1,
                        )
                    )
                )

            else:

                vals.append(
                    100.0
                    * (
                        1.0
                        - 0.5
                        * (
                            self.tp[i]
                            / pos
                            +
                            self.tn[i]
                            / neg
                        )
                    )
                )

        if len(vals) == 0:
            return 0.0

        return float(
            np.mean(vals)
        )


    # ========================================================
    # Report
    # ========================================================

    def report(self):

        return (
            "thr:{:.2f}, "
            "IOU_fg:{:.3f}, "
            "IOU_bg:{:.3f}, "
            "mIoU:{:.3f}, "
            "IOU_std:{:.3f}, "
            "F0.3:{:.3f}, "
            "F0.5:{:.3f}, "
            "F0.7:{:.3f}, "
            "MAE:{:.3f}, "
            "accuracy:{:.3f}, "
            "BER:{:.3f}"
        ).format(
            self.thr,
            self.compute_fg_iou(),
            self.compute_bg_iou(),
            self.compute_miou_2class(),
            self.compute_iou_std(),
            self.compute_fbeta(
                beta=0.3
            ),
            self.compute_fbeta(
                beta=0.5
            ),
            self.compute_fbeta(
                beta=0.7
            ),
            self.compute_mae(),
            self.accuracy(),
            self.ber(),
        )


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
# Cache path
# ============================================================

def get_cache_path(
    sample
):

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
        "test",
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
# Feature extraction / cache
# ============================================================

def get_features(
    model,
    processor,
    sample,
    device,
):

    cache_path = get_cache_path(
        sample
    )

    # --------------------------------------------------------
    # Cache hit
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
            cached[
                "visual_feature"
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        semantic_feature = (
            cached[
                "semantic_feature"
            ]
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
    # Cache miss
    # --------------------------------------------------------

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

    (
        visual_feature,
        semantic_feature,
    ) = model.extract_features(
        inputs,
        target_image_index=0,
    )

    # --------------------------------------------------------
    # Cache save
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
# Prediction save
# ============================================================

def save_prediction(
    sample,
    pred_np,
    threshold,
):

    scene = sample.get(
        "scene",
        "unknown_scene",
    )

    target_stem = os.path.splitext(
        os.path.basename(
            sample["target_path"]
        )
    )[0]

    reference_stem = os.path.splitext(
        os.path.basename(
            sample["reference_path"]
        )
    )[0]

    scene_dir = os.path.join(
        EVAL_OUTPUT_DIR,
        str(scene),
    )

    os.makedirs(
        scene_dir,
        exist_ok=True,
    )

    pair_name = (
        f"{target_stem}"
        f"__ref_"
        f"{reference_stem}"
    )

    # ========================================================
    # Probability map
    # 0～1 float32
    # ========================================================

    prob_path = os.path.join(
        scene_dir,
        f"{pair_name}_prob.npy",
    )

    np.save(
        prob_path,
        pred_np.astype(
            np.float32
        ),
    )

    # ========================================================
    # Binary mask
    # validation selected threshold を使用
    # ========================================================

    pred_bin = (
        pred_np >= threshold
    ).astype(
        np.uint8
    )

    pred_img = (
        pred_bin * 255
    )

    pred_path = os.path.join(
        scene_dir,
        f"{pair_name}_pred.png",
    )

    Image.fromarray(
        pred_img
    ).save(
        pred_path
    )

    return (
        prob_path,
        pred_path,
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Root
    # ========================================================

    resolved_test_root = (
        resolve_root_arg(
            test_root
        )
    )

    if resolved_test_root is None:

        raise RuntimeError(
            "test_root is None"
        )

    # ========================================================
    # Output
    # ========================================================

    os.makedirs(
        EVAL_OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "mirror_head_baseline_best.pt",
    )

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{checkpoint_path}"
        )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "[info] loading test dataset"
    )

    test_dataset = (
        MVMDPairDataset(
            resolved_test_root
        )
    )

    print()

    print(
        f"[info] test_root       = "
        f"{resolved_test_root}"
    )

    print(
        f"[info] test_samples    = "
        f"{len(test_dataset)}"
    )

    print(
        f"[info] checkpoint      = "
        f"{checkpoint_path}"
    )

    print(
        f"[info] output_dir      = "
        f"{EVAL_OUTPUT_DIR}"
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

    # ========================================================
    # Checkpoint load
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.seg_head.load_state_dict(
        checkpoint[
            "seg_head"
        ]
    )

    model.qwen.eval()
    model.seg_head.eval()

    print(
        f"[info] device          = "
        f"{device}"
    )

    print(
        f"[info] checkpoint epoch= "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # ========================================================
    # Validation selected threshold
    # ========================================================

    selected_threshold = float(
        checkpoint.get(
            "val_best_thresh",
            0.5,
        )
    )

    print(
        f"[info] validation threshold = "
        f"{selected_threshold:.2f}"
    )

    if "val_best_iou" in checkpoint:

        print(
            f"[info] val best IoU   = "
            f"{checkpoint['val_best_iou']:.4f}"
        )

    if "val_iou_05" in checkpoint:

        print(
            f"[info] val IoU@0.5    = "
            f"{checkpoint['val_iou_05']:.4f}"
        )

    # ========================================================
    # Thresholds
    #
    # 0.1 ～ 0.9 を 0.1刻み
    # ========================================================

    eval_thresholds = [
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

    metrics_by_thr = {
        th: Metrics(
            thr=th
        )
        for th in eval_thresholds
    }

    # --------------------------------------------------------
    # validation thresholdが0.1刻み以外だった場合にも対応
    # --------------------------------------------------------

    selected_key = round(
        selected_threshold,
        1,
    )

    if selected_key not in metrics_by_thr:

        metrics_by_thr[
            selected_threshold
        ] = Metrics(
            thr=selected_threshold
        )

    # ========================================================
    # Cache stats
    # ========================================================

    cache_hits = 0
    cache_misses = 0

    # ========================================================
    # Progress bar
    # ========================================================

    pbar = tqdm(
        range(
            len(test_dataset)
        ),
        desc="[Test]",
        dynamic_ncols=True,
        ascii=True,
        mininterval=1.0,
        smoothing=0.1,
    )

    # ========================================================
    # Inference
    # ========================================================

    with torch.no_grad():

        for it, index in enumerate(
            pbar,
            start=1,
        ):

            sample = (
                test_dataset[index]
            )

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
            )

            if from_cache:

                cache_hits += 1

            else:

                cache_misses += 1

            # =================================================
            # Segmentation head
            # =================================================

            logits = model.seg_head(
                visual_feature,
                semantic_feature,
                output_size=output_size,
            )

            # =================================================
            # Probability
            # =================================================

            probs = torch.sigmoid(
                logits
            )

            pred_np = (
                probs[
                    0,
                    0,
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            target_np = (
                mask[
                    0,
                    0,
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            # =================================================
            # Sample name
            # =================================================

            target_name = (
                os.path.basename(
                    sample[
                        "target_path"
                    ]
                )
            )

            reference_name = (
                os.path.basename(
                    sample[
                        "reference_path"
                    ]
                )
            )

            sample_name = (
                f"{target_name}"
                f" <- "
                f"{reference_name}"
            )

            # =================================================
            # 0.1～0.9全部で評価
            # =================================================

            for (
                th,
                metrics,
            ) in metrics_by_thr.items():

                metrics.update(
                    pred=pred_np,
                    target=target_np,
                    name=sample_name,
                )

            # =================================================
            # Prediction save
            #
            # PNGはvalidation thresholdで保存
            # NPYはprobabilityなのでthreshold非依存
            # =================================================

            save_prediction(
                sample=sample,
                pred_np=pred_np,
                threshold=selected_threshold,
            )

            # =================================================
            # Progress display
            #
            # validation selected thresholdの値を表示
            # =================================================

            main_metrics = (
                metrics_by_thr[
                    selected_key
                ]
                if selected_key
                in metrics_by_thr
                else metrics_by_thr[
                    selected_threshold
                ]
            )

            if (
                it == 1
                or it % 10 == 0
                or it == len(
                    test_dataset
                )
            ):

                pbar.set_postfix(
                    iou=
                    f"{main_metrics.compute_fg_iou():.4f}",

                    miou=
                    f"{main_metrics.compute_miou_2class():.4f}",

                    mae=
                    f"{main_metrics.compute_mae():.4f}",

                    cache=
                    f"{cache_hits}/{it}",
                )

            # =================================================
            # Cleanup
            # =================================================

            del visual_feature
            del semantic_feature
            del logits
            del probs
            del mask

    # ========================================================
    # Validation-selected result
    # ========================================================

    main_metrics = (
        metrics_by_thr[
            selected_key
        ]
        if selected_key
        in metrics_by_thr
        else metrics_by_thr[
            selected_threshold
        ]
    )

    print()
    print(
        "========================================"
    )

    print(
        "Qwen3-VL Mirror Segmentation Test"
    )

    print(
        "========================================"
    )

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Val threshold    : "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Samples          : "
        f"{len(test_dataset)}"
    )

    print(
        f"Cache hit/miss   : "
        f"{cache_hits}/"
        f"{cache_misses}"
    )

    print(
        f"Prediction dir   : "
        f"{EVAL_OUTPUT_DIR}"
    )

    print()

    print(
        "Official result "
        "(validation-selected threshold)"
    )

    print(
        main_metrics.report()
    )

    # ========================================================
    # Threshold sweep
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "IoU by Threshold"
    )

    print(
        "========================================"
    )

    best_test_iou = -1.0
    best_test_thresh = None

    for th in eval_thresholds:

        metrics = (
            metrics_by_thr[
                th
            ]
        )

        fg_iou = (
            metrics.compute_fg_iou()
        )

        bg_iou = (
            metrics.compute_bg_iou()
        )

        miou = (
            metrics.compute_miou_2class()
        )

        marker = ""

        if abs(
            th
            - selected_threshold
        ) < 1e-8:

            marker = (
                "  <- validation selected"
            )

        print(
            f"th={th:.1f} | "
            f"IOU_fg={fg_iou:.6f} | "
            f"IOU_bg={bg_iou:.6f} | "
            f"mIoU={miou:.6f}"
            f"{marker}"
        )

        if fg_iou > best_test_iou:

            best_test_iou = (
                fg_iou
            )

            best_test_thresh = th

    # ========================================================
    # Test best
    #
    # あくまで参考値
    # ========================================================

    print()

    print(
        f"Validation selected threshold : "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Test best threshold "
        f"(reference only)       : "
        f"{best_test_thresh:.2f}"
    )

    print(
        f"Test best IOU_fg "
        f"(reference only)       : "
        f"{best_test_iou:.6f}"
    )

    # ========================================================
    # Detailed official result
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "Detailed Official Result"
    )

    print(
        "========================================"
    )

    print(
        f"IOU_fg   : "
        f"{main_metrics.compute_fg_iou():.6f}"
    )

    print(
        f"IOU_bg   : "
        f"{main_metrics.compute_bg_iou():.6f}"
    )

    print(
        f"mIoU     : "
        f"{main_metrics.compute_miou_2class():.6f}"
    )

    print(
        f"IOU_std  : "
        f"{main_metrics.compute_iou_std():.6f}"
    )

    print(
        f"F0.3     : "
        f"{main_metrics.compute_fbeta(beta=0.3):.6f}"
    )

    print(
        f"F0.5     : "
        f"{main_metrics.compute_fbeta(beta=0.5):.6f}"
    )

    print(
        f"F0.7     : "
        f"{main_metrics.compute_fbeta(beta=0.7):.6f}"
    )

    print(
        f"MAE      : "
        f"{main_metrics.compute_mae():.6f}"
    )

    print(
        f"Accuracy : "
        f"{main_metrics.accuracy():.6f}"
    )

    print(
        f"BER      : "
        f"{main_metrics.ber():.6f}"
    )


if __name__ == "__main__":
    main()