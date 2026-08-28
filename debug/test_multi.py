from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_name = "Qwen/Qwen3-VL-8B-Instruct"

image1_path = "/data1/nakaue/MAGI-MD/dataset/MVMD/original/mvmd_test_2/diningroom5/pair/pair_10052_10048/JPEGImages_pair/10048.jpg"
image2_path = "/data1/nakaue/MAGI-MD/dataset/MVMD/original/mvmd_test_2/diningroom5/pair/pair_10052_10048/JPEGImages_pair/10052.jpg"
prompt = "この2枚の画像で、視点変化によるものとは異なる変化が起きているところに鏡があります。どこに鏡があるかわかる？"
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_name)

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image1_path},
            {"type": "image", "image": image2_path},
            {"type": "text", "text": prompt},
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

inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

print("Generating response...")
generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n===== Prompt =====")
print(prompt)
print("\n===== Response =====")
print(output_text[0])