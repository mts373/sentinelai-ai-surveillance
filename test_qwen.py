import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


print("=" * 70)
print("SENTINELAI - QWEN2.5-VL MODEL TEST")
print("=" * 70)

print()
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Loading Qwen2.5-VL...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print()
print("Model loaded successfully.")

print(
    "Model device:",
    next(model.parameters()).device
)

print(
    "Model dtype:",
    next(model.parameters()).dtype
)

print()
print(
    "Processor loaded successfully."
)

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated(0)
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(0)
        / 1024**3
    )

    print()
    print(
        f"GPU memory allocated: "
        f"{allocated:.2f} GB"
    )

    print(
        f"GPU memory reserved: "
        f"{reserved:.2f} GB"
    )

print()
print("=" * 70)
print("✅ QWEN MODEL LOAD TEST PASSED")
print("=" * 70)