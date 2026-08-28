# coding: utf-8
"""
analyze_consistency.py

同じ test pair について、
1) 先生のプロンプトをそのまま Qwen3-VL に入力して文章回答を生成
2) 既存の eval_outputs/*/*_prob.npy から Head の鏡マスクを復元
3) GT と比較して IoU を計算
4) 「Qwen の文章上の判断」と「Head のマスク」の整合性を CSV に保存

- Head の再学習はしません。
- 既存の eval_outputs の probability map を使います。
- 先生のプロンプトは変更しません。
- 曖昧な文章回答は UNKNOWN にします。
- 途中で止まっても responses.jsonl から再開できます。
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from config import MODEL_NAME, test_root
from dataset import MVMDPairDataset
from model import QwenMirrorSegmentation


TEACHER_PROMPT = (
    "この2枚の画像で、視点変化によるものとは異なる変化が起きているところに鏡があります。"
    "どこに鏡があるかわかる？"
)


def resolve_root_arg(root_like):
    if root_like is None:
        return None
    if isinstance(root_like, (tuple, list)):
        if len(root_like) == 0:
            return None
        return os.path.expanduser(str(root_like[0]))
    return os.path.expanduser(str(root_like))


def stem_from_path(path):
    return Path(path).stem


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_inputs(processor, target_path, reference_path, prompt, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": target_path},
                {"type": "image", "image": reference_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


@torch.no_grad()
def generate_response(
    qwen,
    processor,
    target_path,
    reference_path,
    prompt,
    device,
    max_new_tokens=256,
):
    inputs = prepare_inputs(
        processor,
        target_path,
        reference_path,
        prompt,
        device,
    )

    generated_ids = qwen.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_ids = inputs["input_ids"]
    trimmed_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return response


def make_pair_key(scene, target_stem, reference_stem):
    return f"{scene}||{target_stem}||{reference_stem}"


def load_response_cache(jsonl_path):
    cache = {}
    if not os.path.exists(jsonl_path):
        return cache

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            key = obj.get("pair_key")
            if key is not None:
                cache[key] = obj

    return cache


def append_response_cache(jsonl_path, obj):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


NEGATIVE_MIRROR_PATTERNS = [
    r"鏡(?:は|が)?(?:ありません|ないです|ない|存在しません|存在しない)",
    r"鏡(?:は|が)?(?:見当たりません|見当たらない)",
    r"鏡(?:は|が)?(?:写っていません|映っていません|見えません)",
    r"鏡(?:を)?確認できません",
    r"鏡(?:を)?確認することができません",
    r"鏡(?:を)?特定できません",
    r"鏡(?:を)?特定することができません",
    r"mirror\s*(?:is\s*)?(?:not\s+present|absent|not\s+visible)",
    r"no\s+mirror",
]

POSITIVE_MIRROR_PATTERNS = [
    r"鏡(?:は|が|の位置は|が写っている場所は|がある場所は)",
    r"鏡があります",
    r"鏡がある",
    r"鏡です",
    r"mirror\s+is\s+(?:located|on|at|in)",
    r"there\s+is\s+(?:a\s+)?mirror",
]


def classify_text_mirror(response):
    text = response.strip()
    low = text.lower()

    for pat in NEGATIVE_MIRROR_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            return "NO"

    for pat in POSITIVE_MIRROR_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            return "YES"

    return "UNKNOWN"


def classify_text_location(response):
    text = response
    low = response.lower()

    if re.search(r"(右側|右の|右端|画像の右|画面の右|右壁|右側の壁)", text):
        return "RIGHT"

    if re.search(r"(左側|左の|左端|画像の左|画面の左|左壁|左側の壁)", text):
        return "LEFT"

    if re.search(r"(中央|中心|真ん中|中央付近)", text):
        return "CENTER"

    if re.search(r"\b(right side|on the right|right side of the image)\b", low):
        return "RIGHT"

    if re.search(r"\b(left side|on the left|left side of the image)\b", low):
        return "LEFT"

    if re.search(r"\b(center|centre|middle)\b", low):
        return "CENTER"

    return "UNKNOWN"


def load_probability(prob_path):
    prob = np.load(prob_path).astype(np.float32)
    prob = np.squeeze(prob)
    return np.clip(prob, 0.0, 1.0)


def load_gt(mask_path):
    gt = np.array(Image.open(mask_path).convert("L")).astype(np.float32)
    if gt.max() > 1.0:
        gt = gt / 255.0
    return gt > 0.5


def compute_iou(pred_bin, gt_bin):
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 1.0

    return float(inter / union)


def compute_mask_information(prob, threshold, min_area_ratio):
    pred_bin = prob >= threshold
    area_ratio = float(pred_bin.mean())

    mask_has_mirror = area_ratio >= min_area_ratio

    if not np.any(pred_bin):
        return {
            "pred_bin": pred_bin,
            "mask_has_mirror": False,
            "mask_nonempty": False,
            "area_ratio": area_ratio,
            "location": "NONE",
            "centroid_x": None,
            "centroid_y": None,
        }

    ys, xs = np.where(pred_bin)
    h, w = pred_bin.shape

    cx = float(xs.mean() / max(w, 1))
    cy = float(ys.mean() / max(h, 1))

    if cx < 1.0 / 3.0:
        location = "LEFT"
    elif cx < 2.0 / 3.0:
        location = "CENTER"
    else:
        location = "RIGHT"

    return {
        "pred_bin": pred_bin,
        "mask_has_mirror": bool(mask_has_mirror),
        "mask_nonempty": True,
        "area_ratio": area_ratio,
        "location": location,
        "centroid_x": cx,
        "centroid_y": cy,
    }


def get_scene_from_sample(sample):
    scene = sample.get("scene")
    if scene is not None:
        return str(scene)

    target_path = Path(sample["target_path"])
    parts = list(target_path.parts)

    if "pair" in parts:
        idx = parts.index("pair")
        if idx > 0:
            return parts[idx - 1]

    return target_path.parent.name


def find_mask_path(sample):
    if "mask_path" in sample:
        p = sample["mask_path"]
        if p is not None and os.path.exists(p):
            return p

    target_path = Path(sample["target_path"])
    target_stem = target_path.stem
    parts = list(target_path.parts)
    candidates = []

    for i, part in enumerate(parts):
        if part == "JPEGImages_pair":
            base = Path(*parts[:i])
            mask_dir = base / "SegmentationClassPNG"
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                candidates.append(str(mask_dir / f"{target_stem}{ext}"))

    scene = get_scene_from_sample(sample)
    test_root_abs = resolve_root_arg(test_root)

    if test_root_abs is not None:
        mask_dir = Path(test_root_abs) / scene / "SegmentationClassPNG"
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            candidates.append(str(mask_dir / f"{target_stem}{ext}"))

    for p in candidates:
        if os.path.exists(p):
            return p

    return None


def find_probability_path(eval_root, scene, target_stem, reference_stem):
    filename = f"{target_stem}__ref_{reference_stem}_prob.npy"
    return os.path.join(eval_root, scene, filename)


def load_threshold(checkpoint_path, manual_threshold):
    if manual_threshold is not None:
        return float(manual_threshold)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            val_best_thresh = ckpt.get("val_best_thresh", None)

            if val_best_thresh is not None:
                print(
                    "[info] threshold from checkpoint "
                    f"val_best_thresh = {float(val_best_thresh):.4f}"
                )
                return float(val_best_thresh)

        except Exception as e:
            print(f"[warn] could not read threshold from checkpoint: {e}")

    print("[warn] threshold could not be resolved; using 0.5")
    return 0.5


def make_consistency_label(text_status, mask_has_mirror):
    if text_status == "UNKNOWN":
        return "UNKNOWN"

    text_yes = text_status == "YES"

    if text_yes == bool(mask_has_mirror):
        return "CONSISTENT"

    return "INCONSISTENT"


def make_case_label(text_status, mask_has_mirror, gt_has_mirror):
    mask_str = "YES" if mask_has_mirror else "NO"
    gt_str = "YES" if gt_has_mirror else "NO"

    return f"TEXT_{text_status}_MASK_{mask_str}_GT_{gt_str}"


CSV_FIELDS = [
    "scene",
    "target",
    "reference",
    "pair_key",
    "response",
    "text_has_mirror",
    "text_location",
    "mask_has_mirror",
    "mask_nonempty",
    "mask_location",
    "mask_area_ratio",
    "mask_centroid_x",
    "mask_centroid_y",
    "gt_has_mirror",
    "gt_area_ratio",
    "iou",
    "consistency",
    "case_label",
    "threshold",
    "min_area_ratio",
    "target_path",
    "reference_path",
    "gt_path",
    "prob_path",
]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--eval-root",
        type=str,
        default="./eval_outputs",
    )

    ap.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/mirror_head_baseline_best.pt",
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    ap.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=0.001,
        help="Headを鏡ありとみなす最小予測面積率。default=0.001",
    )

    ap.add_argument(
        "--output-dir",
        type=str,
        default="./consistency_outputs",
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug用。先頭N pairだけ処理",
    )

    ap.add_argument(
        "--skip-generation",
        action="store_true",
        help="responses.jsonl にあるpairだけ解析する",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    eval_root = os.path.expanduser(args.eval_root)

    checkpoint_path = (
        os.path.expanduser(args.checkpoint)
        if args.checkpoint is not None
        else None
    )

    output_dir = os.path.expanduser(args.output_dir)
    ensure_dir(output_dir)

    response_cache_path = os.path.join(output_dir, "responses.jsonl")
    result_csv_path = os.path.join(output_dir, "consistency_results.csv")
    inconsistent_csv_path = os.path.join(output_dir, "inconsistencies.csv")
    unknown_csv_path = os.path.join(output_dir, "unknown_text_responses.csv")

    threshold = load_threshold(
        checkpoint_path=checkpoint_path,
        manual_threshold=args.threshold,
    )

    print(f"[info] mask threshold    = {threshold:.4f}")
    print(f"[info] min area ratio    = {args.min_mask_area_ratio:.6f}")

    test_root_abs = resolve_root_arg(test_root)

    if test_root_abs is None:
        raise RuntimeError("config.py の test_root が None です。")

    print(f"[info] test_root         = {test_root_abs}")

    test_dataset = MVMDPairDataset(test_root_abs)

    print(f"[info] test samples      = {len(test_dataset)}")

    response_cache = load_response_cache(response_cache_path)

    print(f"[info] cached responses  = {len(response_cache)}")

    processor = None
    model = None
    qwen = None
    device = None

    if not args.skip_generation:
        print("[info] loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        print("[info] loading Qwen...")
        model = QwenMirrorSegmentation()
        qwen = model.qwen
        qwen.eval()

        device = next(qwen.parameters()).device
        print(f"[info] Qwen device       = {device}")

    n_samples = len(test_dataset)

    if args.limit is not None:
        n_samples = min(n_samples, int(args.limit))

    rows = []

    missing_prob = 0
    missing_gt = 0
    skipped_no_response = 0
    generated_count = 0
    reused_response_count = 0

    for index in tqdm(
        range(n_samples),
        desc="Consistency analysis",
        dynamic_ncols=True,
    ):
        sample = test_dataset[index]

        target_path = sample["target_path"]
        reference_path = sample["reference_path"]

        target_stem = stem_from_path(target_path)
        reference_stem = stem_from_path(reference_path)

        scene = get_scene_from_sample(sample)

        pair_key = make_pair_key(
            scene,
            target_stem,
            reference_stem,
        )

        prob_path = find_probability_path(
            eval_root,
            scene,
            target_stem,
            reference_stem,
        )

        if not os.path.exists(prob_path):
            missing_prob += 1
            continue

        gt_path = find_mask_path(sample)

        if gt_path is None or not os.path.exists(gt_path):
            missing_gt += 1
            continue

        if pair_key in response_cache:
            response = response_cache[pair_key]["response"]
            reused_response_count += 1

        else:
            if args.skip_generation:
                skipped_no_response += 1
                continue

            response = generate_response(
                qwen=qwen,
                processor=processor,
                target_path=target_path,
                reference_path=reference_path,
                prompt=TEACHER_PROMPT,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            cache_obj = {
                "pair_key": pair_key,
                "scene": scene,
                "target": target_stem,
                "reference": reference_stem,
                "prompt": TEACHER_PROMPT,
                "response": response,
            }

            append_response_cache(
                response_cache_path,
                cache_obj,
            )

            response_cache[pair_key] = cache_obj
            generated_count += 1

        text_status = classify_text_mirror(response)
        text_location = classify_text_location(response)

        prob = load_probability(prob_path)

        mask_info = compute_mask_information(
            prob=prob,
            threshold=threshold,
            min_area_ratio=args.min_mask_area_ratio,
        )

        pred_bin = mask_info["pred_bin"]

        gt_bin = load_gt(gt_path)

        if pred_bin.shape != gt_bin.shape:
            raise ValueError(
                "shape mismatch: "
                f"pred={pred_bin.shape}, "
                f"gt={gt_bin.shape}, "
                f"pair={pair_key}"
            )

        gt_has_mirror = bool(np.any(gt_bin))
        gt_area_ratio = float(gt_bin.mean())

        iou = compute_iou(pred_bin, gt_bin)

        consistency = make_consistency_label(
            text_status=text_status,
            mask_has_mirror=mask_info["mask_has_mirror"],
        )

        case_label = make_case_label(
            text_status=text_status,
            mask_has_mirror=mask_info["mask_has_mirror"],
            gt_has_mirror=gt_has_mirror,
        )

        rows.append(
            {
                "scene": scene,
                "target": target_stem,
                "reference": reference_stem,
                "pair_key": pair_key,
                "response": response,
                "text_has_mirror": text_status,
                "text_location": text_location,
                "mask_has_mirror": (
                    "YES" if mask_info["mask_has_mirror"] else "NO"
                ),
                "mask_nonempty": (
                    "YES" if mask_info["mask_nonempty"] else "NO"
                ),
                "mask_location": mask_info["location"],
                "mask_area_ratio": mask_info["area_ratio"],
                "mask_centroid_x": mask_info["centroid_x"],
                "mask_centroid_y": mask_info["centroid_y"],
                "gt_has_mirror": "YES" if gt_has_mirror else "NO",
                "gt_area_ratio": gt_area_ratio,
                "iou": iou,
                "consistency": consistency,
                "case_label": case_label,
                "threshold": threshold,
                "min_area_ratio": args.min_mask_area_ratio,
                "target_path": target_path,
                "reference_path": reference_path,
                "gt_path": gt_path,
                "prob_path": prob_path,
            }
        )

    write_csv(result_csv_path, rows)

    inconsistent_rows = [
        row for row in rows
        if row["consistency"] == "INCONSISTENT"
    ]

    unknown_rows = [
        row for row in rows
        if row["text_has_mirror"] == "UNKNOWN"
    ]

    write_csv(inconsistent_csv_path, inconsistent_rows)
    write_csv(unknown_csv_path, unknown_rows)

    consistency_counter = Counter(
        row["consistency"] for row in rows
    )

    case_counter = Counter(
        row["case_label"] for row in rows
    )

    text_counter = Counter(
        row["text_has_mirror"] for row in rows
    )

    print()
    print("============================================================")
    print("Consistency analysis summary")
    print("============================================================")
    print(f"processed              : {len(rows)}")
    print(f"generated responses    : {generated_count}")
    print(f"reused responses       : {reused_response_count}")
    print(f"missing probability    : {missing_prob}")
    print(f"missing GT             : {missing_gt}")
    print(f"skipped(no response)   : {skipped_no_response}")
    print()

    print("Text judgement:")
    for key in ["YES", "NO", "UNKNOWN"]:
        print(f"  {key:8s}: {text_counter.get(key, 0)}")

    print()
    print("Text vs Head:")
    for key in ["CONSISTENT", "INCONSISTENT", "UNKNOWN"]:
        print(f"  {key:12s}: {consistency_counter.get(key, 0)}")

    print()
    print("Interesting cases:")

    interesting_keys = [
        "TEXT_NO_MASK_YES_GT_YES",
        "TEXT_YES_MASK_NO_GT_YES",
        "TEXT_YES_MASK_YES_GT_YES",
        "TEXT_NO_MASK_NO_GT_YES",
        "TEXT_UNKNOWN_MASK_YES_GT_YES",
    ]

    for key in interesting_keys:
        print(f"  {key:30s}: {case_counter.get(key, 0)}")

    print()
    print(f"[saved] all results       : {result_csv_path}")
    print(f"[saved] inconsistencies   : {inconsistent_csv_path}")
    print(f"[saved] unknown responses : {unknown_csv_path}")
    print(f"[saved] response cache    : {response_cache_path}")
    print("============================================================")


if __name__ == "__main__":
    main()