from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

IMAGE1 = "/data1/nakaue/MAGI-MD/dataset/MVMD/original/mvmd_test_2/room3/pair/pair_0000_0001/JPEGImages_pair/0000.jpg"
IMAGE2 = "/data1/nakaue/MAGI-MD/dataset/MVMD/original/mvmd_test_2/room3/pair/pair_0000_0001/JPEGImages_pair/0001.jpg"

PROMPT = (
    "Compare these two views of the same scene. "
    "A mirror is a reflective physical surface whose reflected appearance "
    "may change with viewpoint. Identify the mirror region."
)


def main():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE1},
                {"type": "image", "image": IMAGE2},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    print("Preparing inputs...")

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

    print("\n===== Input keys =====")
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value)}")

    print("\nRunning forward pass...")

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    print("\n===== Output type =====")
    print(type(outputs))

    print("\n===== Output keys =====")
    if hasattr(outputs, "keys"):
        print(outputs.keys())

    print("\n===== Logits =====")
    if hasattr(outputs, "logits"):
        print("logits shape:", tuple(outputs.logits.shape))

    print("\n===== Hidden states =====")

    hidden_states = getattr(outputs, "hidden_states", None)

    if hidden_states is None:
        print("hidden_states is None")
    else:
        print("number of hidden-state layers:", len(hidden_states))

        for i, h in enumerate(hidden_states):
            print(
                f"layer {i}: "
                f"shape={tuple(h.shape)}, "
                f"dtype={h.dtype}"
            )

        last_hidden = hidden_states[-1]

        print("\nLast hidden state:")
        print("shape:", tuple(last_hidden.shape))

    print("\n===== Token inspection =====")

    input_ids = inputs["input_ids"][0]

    print("number of input tokens:", input_ids.shape[0])

    tokenizer = processor.tokenizer

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.detach().cpu().tolist()
    )

    for i, token in enumerate(tokens):
        if "vision" in token.lower() or "image" in token.lower():
            print(i, token)

    print("\n===== Model structure =====")

    print("Top-level modules:")
    for name, module in model.named_children():
        print(name, "->", type(module).__name__)

    print("\nSearching for vision-related modules:")

    for name, module in model.named_modules():
        lname = name.lower()

        if any(
            keyword in lname
            for keyword in [
                "visual",
                "vision",
                "merger",
            ]
        ):
            print(name, "->", type(module).__name__)

    print("image_grid_thw:")
    print(inputs["image_grid_thw"])

    print("\nDone.")



if __name__ == "__main__":
    main()