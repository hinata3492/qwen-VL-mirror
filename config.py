# coding: utf-8

# =========================
# Model
# =========================

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


# =========================
# Dataset
# =========================

DATA_ROOT = "/data1/nakaue/MAGI-MD/dataset"

training_root = f"{DATA_ROOT}/MIXED2/MVMD_MAGI_MIX"
validation_root = f"{DATA_ROOT}/MVMD/original/mvmd_val_30"
test_root = f"{DATA_ROOT}/MVMD/original/mvmd_test_2"


# =========================
# Prompt
# =========================

PROMPT = "この2枚の画像で、視点変化によるものとは異なる変化が起きているところに鏡があります。どこに鏡があるかわかる？"

# =========================
# Training
# =========================

NUM_EPOCHS = 50

# debug時の1e-4では多少振れていたので少し下げる
LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

BATCH_SIZE = 1

SEED = 42

FEATURE_CACHE_DIR = "./feature_cache"

# =========================
# Segmentation Head
# =========================

QWEN_HIDDEN_DIM = 4096

SEG_HIDDEN_DIM = 256


# =========================
# Output
# =========================

CHECKPOINT_DIR = "./checkpoints"

OUTPUT_DIR = "./outputs"

RESULT_DIR = "./results"