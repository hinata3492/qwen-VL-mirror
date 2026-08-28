# coding: utf-8
"""
analyze_feature_alignment.py

目的:
既存の consistency_outputs/consistency_results.csv から代表ケースを抽出し、
現在 Head が使っている prompt final feature と、
Qwen が実際に文章生成時に使う hidden features の cosine similarity を比較する。

比較対象:
- prompt final token feature
- first generated token feature
- answer mean feature
- 「鏡」/ "mirror" を含む生成tokenの平均feature

出力:
feature_alignment_outputs/
  feature_alignment.csv
  summary.txt
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from config import MODEL_NAME
from model import QwenMirrorSegmentation


TEACHER_PROMPT = (
    "この2枚の画像で、視点変化によるものとは異なる変化が起きているところに鏡があります。"
    "どこに鏡があるかわかる？"
)


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


def build_messages(target_path, reference_path, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": target_path},
                {"type": "image", "image": reference_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_inputs(processor, target_path, reference_path, prompt, device):
    messages = build_messages(target_path, reference_path, prompt)

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
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }


def get_last_hidden(outputs):
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]

    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state

    raise RuntimeError("hidden state was not found in Qwen output")


@torch.no_grad()
def extract_prompt_final_feature(qwen, inputs):
    outputs = qwen(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    hidden = get_last_hidden(outputs)

    return hidden[0, -1, :].float()


@torch.no_grad()
def generate_with_hidden_features(
    qwen,
    processor,
    inputs,
    max_new_tokens=256,
):
    generated = qwen.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )

    sequences = generated.sequences
    prompt_len = inputs["input_ids"].shape[1]
    generated_token_ids = sequences[0, prompt_len:]

    response = processor.decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    hidden_steps = generated.hidden_states

    if hidden_steps is None:
        raise RuntimeError("generate() returned no hidden_states")

    token_features = []

    for step_hidden in hidden_steps:
        if step_hidden is None:
            continue

        if isinstance(step_hidden, (tuple, list)):
            last_layer = step_hidden[-1]
        else:
            last_layer = step_hidden

        if last_layer is None or last_layer.ndim != 3:
            continue

        token_features.append(
            last_layer[0, -1, :].float()
        )

    n_gen = int(generated_token_ids.numel())

    if len(token_features) > n_gen:
        token_features = token_features[-n_gen:]

    if len(token_features) == 0:
        raise RuntimeError("could not extract generated-token hidden features")

    aligned_n = min(len(token_features), n_gen)

    token_features = token_features[:aligned_n]
    token_ids_aligned = generated_token_ids[:aligned_n]

    stacked = torch.stack(token_features, dim=0)

    first_generated = stacked[0]
    answer_mean = stacked.mean(dim=0)

    mirror_indices = []
    mirror_token_texts = []

    for i, tok_id in enumerate(token_ids_aligned.tolist()):
        tok_text = processor.decode(
            [tok_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        low = tok_text.lower()

        if ("鏡" in tok_text) or ("mirror" in low):
            mirror_indices.append(i)
            mirror_token_texts.append(tok_text)

    mirror_feature = None

    if len(mirror_indices) > 0:
        mirror_feature = stacked[mirror_indices].mean(dim=0)

    return {
        "response": response,
        "generated_token_count": aligned_n,
        "first_generated_feature": first_generated,
        "answer_mean_feature": answer_mean,
        "mirror_feature": mirror_feature,
        "mirror_token_found": len(mirror_indices) > 0,
        "mirror_token_text": "|".join(mirror_token_texts),
    }


def cosine(a, b):
    return float(
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=-1,
        ).item()
    )


DEFAULT_CASES = [
    "TEXT_NO_MASK_YES_GT_YES",
    "TEXT_YES_MASK_NO_GT_YES",
    "TEXT_YES_MASK_YES_GT_YES",
]


def sample_rows_by_case(rows, cases, per_case):
    grouped = defaultdict(list)

    for row in rows:
        case = row.get("case_label", "")

        if case in cases:
            grouped[case].append(row)

    selected = []

    for case in cases:
        selected.extend(
            grouped.get(case, [])[:per_case]
        )

    return selected


def safe_mean(vals):
    vals = [
        v for v in vals
        if v is not None and not np.isnan(v)
    ]

    if len(vals) == 0:
        return None

    return float(np.mean(vals))


def safe_std(vals):
    vals = [
        v for v in vals
        if v is not None and not np.isnan(v)
    ]

    if len(vals) == 0:
        return None

    return float(np.std(vals))


def make_summary(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["case_label"]].append(row)

    lines = [
        "Feature alignment summary",
        "=" * 60,
    ]

    for case, bucket in grouped.items():
        sim_first = [
            float(x["cos_prompt_vs_first"])
            for x in bucket
            if x["cos_prompt_vs_first"] != ""
        ]

        sim_mean = [
            float(x["cos_prompt_vs_answer_mean"])
            for x in bucket
            if x["cos_prompt_vs_answer_mean"] != ""
        ]

        sim_mirror = [
            float(x["cos_prompt_vs_mirror"])
            for x in bucket
            if x["cos_prompt_vs_mirror"] != ""
        ]

        lines.append("")
        lines.append(f"[{case}] n={len(bucket)}")
        lines.append(
            "  prompt vs first generated : "
            f"mean={safe_mean(sim_first)} std={safe_std(sim_first)}"
        )
        lines.append(
            "  prompt vs answer mean     : "
            f"mean={safe_mean(sim_mean)} std={safe_std(sim_mean)}"
        )
        lines.append(
            "  prompt vs mirror token    : "
            f"mean={safe_mean(sim_mirror)} std={safe_std(sim_mirror)} n={len(sim_mirror)}"
        )

    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--consistency-csv",
        type=str,
        default="./consistency_outputs/consistency_results.csv",
    )

    ap.add_argument(
        "--output-dir",
        type=str,
        default="./feature_alignment_outputs",
    )

    ap.add_argument(
        "--per-case",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
    )

    return ap.parse_args()


def main():
    args = parse_args()

    consistency_csv = os.path.expanduser(
        args.consistency_csv
    )

    output_dir = os.path.expanduser(
        args.output_dir
    )

    ensure_dir(output_dir)

    rows = load_csv(consistency_csv)

    selected = sample_rows_by_case(
        rows,
        args.cases,
        args.per_case,
    )

    print(f"[info] source rows       = {len(rows)}")
    print(f"[info] selected rows     = {len(selected)}")

    for case in args.cases:
        n = sum(
            1 for r in selected
            if r.get("case_label") == case
        )

        print(f"[info] {case:30s} = {n}")

    print("[info] loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("[info] loading model...")
    model = QwenMirrorSegmentation()

    qwen = model.qwen
    qwen.eval()

    device = next(qwen.parameters()).device
    print(f"[info] device            = {device}")

    output_rows = []

    for row in tqdm(
        selected,
        desc="Feature alignment",
        dynamic_ncols=True,
    ):
        target_path = row["target_path"]
        reference_path = row["reference_path"]

        inputs = prepare_inputs(
            processor,
            target_path,
            reference_path,
            TEACHER_PROMPT,
            device,
        )

        prompt_final = extract_prompt_final_feature(
            qwen,
            inputs,
        )

        gen = generate_with_hidden_features(
            qwen,
            processor,
            inputs,
            max_new_tokens=args.max_new_tokens,
        )

        sim_first = cosine(
            prompt_final,
            gen["first_generated_feature"],
        )

        sim_mean = cosine(
            prompt_final,
            gen["answer_mean_feature"],
        )

        if gen["mirror_feature"] is not None:
            sim_mirror = cosine(
                prompt_final,
                gen["mirror_feature"],
            )
        else:
            sim_mirror = ""

        output_rows.append(
            {
                "scene": row.get("scene", ""),
                "target": row.get("target", ""),
                "reference": row.get("reference", ""),
                "case_label": row.get("case_label", ""),
                "text_has_mirror": row.get("text_has_mirror", ""),
                "mask_has_mirror": row.get("mask_has_mirror", ""),
                "gt_has_mirror": row.get("gt_has_mirror", ""),
                "iou": row.get("iou", ""),
                "old_response": row.get("response", ""),
                "new_response": gen["response"],
                "generated_token_count": gen["generated_token_count"],
                "mirror_token_found": (
                    "YES" if gen["mirror_token_found"] else "NO"
                ),
                "mirror_token_text": gen["mirror_token_text"],
                "cos_prompt_vs_first": sim_first,
                "cos_prompt_vs_answer_mean": sim_mean,
                "cos_prompt_vs_mirror": sim_mirror,
                "target_path": target_path,
                "reference_path": reference_path,
            }
        )

    fields = [
        "scene",
        "target",
        "reference",
        "case_label",
        "text_has_mirror",
        "mask_has_mirror",
        "gt_has_mirror",
        "iou",
        "old_response",
        "new_response",
        "generated_token_count",
        "mirror_token_found",
        "mirror_token_text",
        "cos_prompt_vs_first",
        "cos_prompt_vs_answer_mean",
        "cos_prompt_vs_mirror",
        "target_path",
        "reference_path",
    ]

    csv_path = os.path.join(
        output_dir,
        "feature_alignment.csv",
    )

    write_csv(
        csv_path,
        output_rows,
        fields,
    )

    summary = make_summary(output_rows)

    summary_path = os.path.join(
        output_dir,
        "summary.txt",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print(summary)
    print()
    print(f"[saved] {csv_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()