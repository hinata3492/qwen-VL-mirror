# coding: utf-8
"""
analyze_group_correctness.py

目的
----
既存の consistency_outputs/consistency_results.csv を使って、
Text / Head の一致・不一致グループごとに、

- VLM の存在判定が正しいか
- VLM の位置説明が GT と合っているか
- Head の存在判定が正しいか
- Head の segmentation が GT と十分一致しているか (IoU >= threshold)

を集計する。

主なグループ
------------
TEXT_YES_MASK_YES
TEXT_NO_MASK_YES
TEXT_YES_MASK_NO
TEXT_NO_MASK_NO
TEXT_UNKNOWN_MASK_YES
TEXT_UNKNOWN_MASK_NO

出力
----
group_correctness_outputs/
  detailed_results.csv
  group_summary.csv
  summary.txt

注意
----
VLM の「位置正解」は、文章から抽出済みの LEFT/CENTER/RIGHT と、
GT mask の重心から求めた LEFT/CENTER/RIGHT を比較する簡易評価です。

Head の segmentation 正解は default で IoU >= 0.5 とします。
これは分析用の閾値であり、IoUそのものもCSVに残します。
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image


# ============================================================
# IO
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# helpers
# ============================================================

def to_bool_yes_no(value):
    return str(value).strip().upper() == "YES"


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def load_gt_mask(gt_path):
    img = Image.open(gt_path).convert("L")
    gt = np.asarray(img).astype(np.float32)

    if gt.max() > 1.0:
        gt = gt / 255.0

    return gt > 0.5


def get_mask_location(mask):
    """
    mask重心のx座標で LEFT/CENTER/RIGHT を決める。
    GTに鏡がなければ NONE。
    """
    if not np.any(mask):
        return "NONE"

    ys, xs = np.where(mask)
    width = mask.shape[1]

    cx = float(xs.mean() / max(width, 1))

    if cx < 1.0 / 3.0:
        return "LEFT"
    elif cx < 2.0 / 3.0:
        return "CENTER"
    else:
        return "RIGHT"


def get_group(text_status, mask_has_mirror):
    text_status = str(text_status).strip().upper()
    mask_status = "YES" if mask_has_mirror else "NO"

    return f"TEXT_{text_status}_MASK_{mask_status}"


# ============================================================
# correctness
# ============================================================

def evaluate_vlm_presence(text_status, gt_has_mirror):
    """
    YES/NO がGT presenceと一致するか。
    UNKNOWNは None。
    """
    text_status = str(text_status).strip().upper()

    if text_status == "UNKNOWN":
        return None

    text_has_mirror = text_status == "YES"
    return bool(text_has_mirror == gt_has_mirror)


def evaluate_vlm_location(
    text_status,
    text_location,
    gt_has_mirror,
    gt_location,
):
    """
    位置まで含めた簡易正解判定。

    GTあり:
      Text YES かつ LEFT/CENTER/RIGHT がGTと一致 -> True
      Text YES だが location UNKNOWN -> None
      Text NO -> False
      Text UNKNOWN -> None

    GTなし:
      Text NO -> True
      Text YES -> False
      Text UNKNOWN -> None
    """
    text_status = str(text_status).strip().upper()
    text_location = str(text_location).strip().upper()

    if not gt_has_mirror:
        if text_status == "NO":
            return True
        if text_status == "YES":
            return False
        return None

    # GTに鏡あり
    if text_status == "NO":
        return False

    if text_status == "UNKNOWN":
        return None

    if text_location not in {"LEFT", "CENTER", "RIGHT"}:
        return None

    return bool(text_location == gt_location)


def evaluate_head_presence(mask_has_mirror, gt_has_mirror):
    return bool(mask_has_mirror == gt_has_mirror)


def evaluate_head_segmentation(iou, gt_has_mirror, iou_threshold):
    """
    GTに鏡ありなら IoU >= threshold を segmentation correct とする。
    GTに鏡なしなら、IoU==1 かつ maskなしが理想だが、
    consistency CSV の iou が 1.0 なら True とする。
    """
    if np.isnan(iou):
        return None

    if gt_has_mirror:
        return bool(iou >= iou_threshold)

    return bool(iou >= 1.0 - 1e-12)


def bool_to_text(value):
    if value is None:
        return "UNKNOWN"
    return "YES" if value else "NO"


# ============================================================
# summary
# ============================================================

def ratio(numer, denom):
    if denom == 0:
        return None
    return numer / denom


def fmt_ratio(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def summarize_group(rows):
    n = len(rows)

    # VLM presence
    vlm_presence_known = [
        r for r in rows
        if r["vlm_presence_correct"] != "UNKNOWN"
    ]
    vlm_presence_correct = sum(
        r["vlm_presence_correct"] == "YES"
        for r in vlm_presence_known
    )

    # VLM location
    vlm_location_known = [
        r for r in rows
        if r["vlm_location_correct"] != "UNKNOWN"
    ]
    vlm_location_correct = sum(
        r["vlm_location_correct"] == "YES"
        for r in vlm_location_known
    )

    # Head presence
    head_presence_known = [
        r for r in rows
        if r["head_presence_correct"] != "UNKNOWN"
    ]
    head_presence_correct = sum(
        r["head_presence_correct"] == "YES"
        for r in head_presence_known
    )

    # Head segmentation
    head_seg_known = [
        r for r in rows
        if r["head_segmentation_correct"] != "UNKNOWN"
    ]
    head_seg_correct = sum(
        r["head_segmentation_correct"] == "YES"
        for r in head_seg_known
    )

    ious = [
        safe_float(r["iou"])
        for r in rows
    ]
    ious = [
        x for x in ious
        if not np.isnan(x)
    ]

    return {
        "n": n,

        "vlm_presence_correct_n": vlm_presence_correct,
        "vlm_presence_evaluable_n": len(vlm_presence_known),
        "vlm_presence_accuracy": ratio(
            vlm_presence_correct,
            len(vlm_presence_known),
        ),

        "vlm_location_correct_n": vlm_location_correct,
        "vlm_location_evaluable_n": len(vlm_location_known),
        "vlm_location_accuracy": ratio(
            vlm_location_correct,
            len(vlm_location_known),
        ),

        "head_presence_correct_n": head_presence_correct,
        "head_presence_evaluable_n": len(head_presence_known),
        "head_presence_accuracy": ratio(
            head_presence_correct,
            len(head_presence_known),
        ),

        "head_segmentation_correct_n": head_seg_correct,
        "head_segmentation_evaluable_n": len(head_seg_known),
        "head_segmentation_accuracy": ratio(
            head_seg_correct,
            len(head_seg_known),
        ),

        "mean_iou": (
            float(np.mean(ious))
            if len(ious) > 0
            else None
        ),

        "median_iou": (
            float(np.median(ious))
            if len(ious) > 0
            else None
        ),
    }


# ============================================================
# main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-csv",
        type=str,
        default="./consistency_outputs/consistency_results.csv",
    )

    ap.add_argument(
        "--output-dir",
        type=str,
        default="./group_correctness_outputs",
    )

    ap.add_argument(
        "--head-iou-threshold",
        type=float,
        default=0.5,
        help="Head segmentationを正解とみなすIoU閾値",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    input_csv = os.path.expanduser(
        args.input_csv
    )

    output_dir = os.path.expanduser(
        args.output_dir
    )

    ensure_dir(output_dir)

    rows = load_csv(input_csv)

    print(f"[info] input rows         = {len(rows)}")
    print(
        f"[info] head IoU threshold = "
        f"{args.head_iou_threshold:.3f}"
    )

    detailed_rows = []

    for row in rows:
        gt_path = row.get("gt_path", "")

        if not gt_path or not os.path.exists(gt_path):
            print(f"[warn] GT not found: {gt_path}")
            continue

        gt_mask = load_gt_mask(gt_path)

        gt_has_mirror = bool(
            np.any(gt_mask)
        )

        gt_location = get_mask_location(
            gt_mask
        )

        text_status = row.get(
            "text_has_mirror",
            "UNKNOWN",
        ).strip().upper()

        text_location = row.get(
            "text_location",
            "UNKNOWN",
        ).strip().upper()

        mask_has_mirror = to_bool_yes_no(
            row.get(
                "mask_has_mirror",
                "NO",
            )
        )

        iou = safe_float(
            row.get(
                "iou",
                "",
            )
        )

        vlm_presence_correct = evaluate_vlm_presence(
            text_status=text_status,
            gt_has_mirror=gt_has_mirror,
        )

        vlm_location_correct = evaluate_vlm_location(
            text_status=text_status,
            text_location=text_location,
            gt_has_mirror=gt_has_mirror,
            gt_location=gt_location,
        )

        head_presence_correct = evaluate_head_presence(
            mask_has_mirror=mask_has_mirror,
            gt_has_mirror=gt_has_mirror,
        )

        head_segmentation_correct = evaluate_head_segmentation(
            iou=iou,
            gt_has_mirror=gt_has_mirror,
            iou_threshold=args.head_iou_threshold,
        )

        group = get_group(
            text_status=text_status,
            mask_has_mirror=mask_has_mirror,
        )

        # 不一致時に「どちらが正しいか」を簡単に示す
        winner = "OTHER"

        if text_status in {"YES", "NO"}:
            vlm_presence_bool = vlm_presence_correct
        else:
            vlm_presence_bool = None

        head_seg_bool = head_segmentation_correct

        if (
            vlm_presence_bool is False
            and head_seg_bool is True
        ):
            winner = "HEAD_CORRECT_VLM_WRONG"

        elif (
            vlm_location_correct is True
            and head_seg_bool is False
        ):
            winner = "VLM_CORRECT_HEAD_WRONG"

        elif (
            vlm_location_correct is True
            and head_seg_bool is True
        ):
            winner = "BOTH_CORRECT"

        elif (
            vlm_presence_bool is False
            and head_seg_bool is False
        ):
            winner = "BOTH_WRONG"

        elif (
            vlm_location_correct is None
        ):
            winner = "VLM_LOCATION_UNKNOWN"

        out = dict(row)

        out.update(
            {
                "group":
                    group,

                "gt_location":
                    gt_location,

                "vlm_presence_correct":
                    bool_to_text(
                        vlm_presence_correct
                    ),

                "vlm_location_correct":
                    bool_to_text(
                        vlm_location_correct
                    ),

                "head_presence_correct":
                    bool_to_text(
                        head_presence_correct
                    ),

                "head_segmentation_correct":
                    bool_to_text(
                        head_segmentation_correct
                    ),

                "head_iou_threshold":
                    args.head_iou_threshold,

                "which_is_correct":
                    winner,
            }
        )

        detailed_rows.append(out)

    # --------------------------------------------------------
    # group summary
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for row in detailed_rows:
        grouped[
            row["group"]
        ].append(
            row
        )

    group_order = [
        "TEXT_YES_MASK_YES",
        "TEXT_NO_MASK_YES",
        "TEXT_YES_MASK_NO",
        "TEXT_NO_MASK_NO",
        "TEXT_UNKNOWN_MASK_YES",
        "TEXT_UNKNOWN_MASK_NO",
    ]

    summary_rows = []

    for group in group_order:
        bucket = grouped.get(
            group,
            []
        )

        if len(bucket) == 0:
            continue

        s = summarize_group(
            bucket
        )

        summary_rows.append(
            {
                "group":
                    group,

                "n":
                    s["n"],

                "vlm_presence_correct_n":
                    s["vlm_presence_correct_n"],

                "vlm_presence_evaluable_n":
                    s["vlm_presence_evaluable_n"],

                "vlm_presence_accuracy":
                    s["vlm_presence_accuracy"],

                "vlm_location_correct_n":
                    s["vlm_location_correct_n"],

                "vlm_location_evaluable_n":
                    s["vlm_location_evaluable_n"],

                "vlm_location_accuracy":
                    s["vlm_location_accuracy"],

                "head_presence_correct_n":
                    s["head_presence_correct_n"],

                "head_presence_evaluable_n":
                    s["head_presence_evaluable_n"],

                "head_presence_accuracy":
                    s["head_presence_accuracy"],

                "head_segmentation_correct_n":
                    s["head_segmentation_correct_n"],

                "head_segmentation_evaluable_n":
                    s["head_segmentation_evaluable_n"],

                "head_segmentation_accuracy":
                    s["head_segmentation_accuracy"],

                "mean_iou":
                    s["mean_iou"],

                "median_iou":
                    s["median_iou"],
            }
        )

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    detailed_path = os.path.join(
        output_dir,
        "detailed_results.csv",
    )

    summary_path = os.path.join(
        output_dir,
        "group_summary.csv",
    )

    text_summary_path = os.path.join(
        output_dir,
        "summary.txt",
    )

    detailed_fields = list(
        detailed_rows[0].keys()
    ) if detailed_rows else []

    write_csv(
        detailed_path,
        detailed_rows,
        detailed_fields,
    )

    summary_fields = [
        "group",
        "n",

        "vlm_presence_correct_n",
        "vlm_presence_evaluable_n",
        "vlm_presence_accuracy",

        "vlm_location_correct_n",
        "vlm_location_evaluable_n",
        "vlm_location_accuracy",

        "head_presence_correct_n",
        "head_presence_evaluable_n",
        "head_presence_accuracy",

        "head_segmentation_correct_n",
        "head_segmentation_evaluable_n",
        "head_segmentation_accuracy",

        "mean_iou",
        "median_iou",
    ]

    write_csv(
        summary_path,
        summary_rows,
        summary_fields,
    )

    # --------------------------------------------------------
    # print + txt
    # --------------------------------------------------------

    lines = []

    lines.append(
        "Group correctness summary"
    )

    lines.append(
        "=" * 90
    )

    lines.append(
        f"Head segmentation correct: IoU >= {args.head_iou_threshold:.3f}"
    )

    lines.append("")

    for row in summary_rows:
        lines.append(
            f"[{row['group']}] n={row['n']}"
        )

        lines.append(
            "  VLM presence accuracy     : "
            f"{fmt_ratio(row['vlm_presence_accuracy'])} "
            f"({row['vlm_presence_correct_n']}/"
            f"{row['vlm_presence_evaluable_n']})"
        )

        lines.append(
            "  VLM location accuracy     : "
            f"{fmt_ratio(row['vlm_location_accuracy'])} "
            f"({row['vlm_location_correct_n']}/"
            f"{row['vlm_location_evaluable_n']})"
        )

        lines.append(
            "  Head presence accuracy    : "
            f"{fmt_ratio(row['head_presence_accuracy'])} "
            f"({row['head_presence_correct_n']}/"
            f"{row['head_presence_evaluable_n']})"
        )

        lines.append(
            "  Head segmentation accuracy: "
            f"{fmt_ratio(row['head_segmentation_accuracy'])} "
            f"({row['head_segmentation_correct_n']}/"
            f"{row['head_segmentation_evaluable_n']})"
        )

        lines.append(
            "  Mean IoU                  : "
            f"{row['mean_iou']}"
        )

        lines.append(
            "  Median IoU                : "
            f"{row['median_iou']}"
        )

        lines.append("")

    # winner count
    winner_counts = defaultdict(int)

    for row in detailed_rows:
        winner_counts[
            row["which_is_correct"]
        ] += 1

    lines.append(
        "Which is correct?"
    )

    for key in [
        "HEAD_CORRECT_VLM_WRONG",
        "VLM_CORRECT_HEAD_WRONG",
        "BOTH_CORRECT",
        "BOTH_WRONG",
        "VLM_LOCATION_UNKNOWN",
        "OTHER",
    ]:
        lines.append(
            f"  {key:28s}: "
            f"{winner_counts.get(key, 0)}"
        )

    summary_text = "\n".join(
        lines
    )

    with open(
        text_summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            summary_text
        )

    print()
    print(
        summary_text
    )

    print()
    print(
        f"[saved] {detailed_path}"
    )

    print(
        f"[saved] {summary_path}"
    )

    print(
        f"[saved] {text_summary_path}"
    )


if __name__ == "__main__":
    main()