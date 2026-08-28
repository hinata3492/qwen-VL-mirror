from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_name = "Qwen/Qwen3-VL-8B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_name)

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loaded successfully.")
print("Model device:", next(model.parameters()).device)
print("Model dtype:", next(model.parameters()).dtype)

if torch.cuda.is_available():
    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB"
    )