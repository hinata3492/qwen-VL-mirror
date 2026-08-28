from pathlib import Path
import csv
import torch

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

DATA_ROOT = Path(
    "/data1/nakaue/MAGI-MD/dataset/MVMD/original/mvmd_test_2"
)

OUTPUT_CSV = Path("qwen3vl_mirror_results.csv")

SKIP_NAMES = {
    "confidence_by_scene_frame_unique",
}

PROMPT = (
    "Compare these two views of the same scene and determine whether a mirror is present. "
    "If a mirror is present, describe its location precisely. "
    "Then explain the visual evidence supporting your decision. "
    "Consider reflection, viewpoint-dependent appearance changes, "
    "and whether the region could instead be a real window or opening. "
    "If there is no mirror, explain why. "
    "Answer in exactly the following format:\n"
    "Mirror: Yes or No\n"
    "Location: ...\n"
    "Evidence: ...\n"
    "Distinction from window/opening: ...\n"
    "Multi-view evidence: ..."
)


def get_one_pair_per_scene(root):
    samples = []

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue

        if scene_dir.name in SKIP_NAMES:
            continue

        pair_root = scene_dir / "pair"

        if not pair_root.exists():
            print(f"[SKIP] {scene_dir.name}: pair directory not found")
            continue

        pair_dirs = sorted(
            p for p in pair_root.iterdir()
            if p.is_dir() and p.name.startswith("pair_")
        )

        if not pair_dirs:
            print(f"[SKIP] {scene_dir.name}: no pairs found")
            continue

        selected_pair = pair_dirs[0]
        image_dir = selected_pair / "JPEGImages_pair"

        images = sorted(image_dir.glob("*.jpg"))

        if len(images) != 2:
            print(
                f"[SKIP] {scene_dir.name}: "
                f"{len(images)} images found in {selected_pair.name}"
            )
            continue

        samples.append(
            {
                "scene": scene_dir.name,
                "pair": selected_pair.name,
                "image1": images[0],
                "image2": images[1],
            }
        )

    return samples


def run_inference(model, processor, image1, image2):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image1)},
                {"type": "image", "image": str(image2)},
                {"type": "text", "text": PROMPT},
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

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


def save_results(results, output_path):
    fieldnames = [
        "scene",
        "pair",
        "image1",
        "image2",
        "response",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def main():
    print("Collecting one pair per scene...")
    samples = get_one_pair_per_scene(DATA_ROOT)

    print(f"Found {len(samples)} scenes.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Model loaded.")

    results = []

    for i, sample in enumerate(samples, start=1):
        print()
        print("=" * 70)
        print(
            f"[{i}/{len(samples)}] "
            f"Scene: {sample['scene']} | "
            f"Pair: {sample['pair']}"
        )
        print(
            f"Images: "
            f"{sample['image1'].name}, "
            f"{sample['image2'].name}"
        )

        try:
            response = run_inference(
                model,
                processor,
                sample["image1"],
                sample["image2"],
            )

            print("\nResponse:")
            print(response)

            results.append(
                {
                    "scene": sample["scene"],
                    "pair": sample["pair"],
                    "image1": str(sample["image1"]),
                    "image2": str(sample["image2"]),
                    "response": response,
                }
            )

        except Exception as e:
            print(f"[ERROR] {e}")

            results.append(
                {
                    "scene": sample["scene"],
                    "pair": sample["pair"],
                    "image1": str(sample["image1"]),
                    "image2": str(sample["image2"]),
                    "response": f"ERROR: {e}",
                }
            )

        # 毎回保存
        save_results(results, OUTPUT_CSV)

        # 念のためGPUキャッシュを解放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("Finished.")
    print(f"Results saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()