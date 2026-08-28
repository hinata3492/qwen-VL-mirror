from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from model import QwenMirrorSegmentation

import torch


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

IMAGE1 = "/data1/nakaue/MAGI-MD/dataset/MVMD/original/mvmd_test_2/room3/pair/pair_0000_0001/JPEGImages_pair/0000.jpg"
IMAGE2 = "/data1/nakaue/MAGI-MD/dataset/MVMD/original/mvmd_test_2/room3/pair/pair_0000_0001/JPEGImages_pair/0001.jpg"

PROMPT = (
    "Compare these two views of the same scene. "
    "Identify the mirror region."
)


def main():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = QwenMirrorSegmentation()

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

    device = next(model.qwen.parameters()).device

    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    model.seg_head = model.seg_head.to(device)

    model.eval()

    with torch.no_grad():
        pred = model(
            inputs,
            target_image_index=0,
            output_size=(480, 640),
        )

    print("prediction shape:", pred.shape)
    print("dtype:", pred.dtype)
    print("min:", pred.min().item())
    print("max:", pred.max().item())

    prob = torch.sigmoid(pred)

    print("prob min:", prob.min().item())
    print("prob max:", prob.max().item())


if __name__ == "__main__":
    main()