import os
import random
import sys
original_stdout = sys.stdout
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Adjust the GPU IDs as needed.
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# Import the shared prompt and tool definitions used by the template.
from visharness.prompts import *
from datasets import load_dataset

from swift import get_model_processor, get_template
from swift.dataset import LazyLLMDataset
from swift.utils import seed_everything
sys.stdout = original_stdout

# ================= Environment setup =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

seed_everything(42)

# ================= Configuration =================
model_id_or_path = '/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/models/LLM/Qwen3-VL-8B-Thinking/'  # Replace with the local model path.
dataset_path = (
    project_root
    / "training_data/SFT/KimiK2.5-Qwen3.5_397B_FP8-Merged-VisionAgent-4K-20260927-patch56k"
    / "merged_sft_data_swift_cmd.jsonl"
)
os.environ["ROOT_IMAGE_DIR"] = str(dataset_path.parent)
max_length = 16000

# ================= Load the dataset =================
print(f"Loading and formatting dataset from {dataset_path} ...", flush=True)
processed_dataset = load_dataset("json", data_files=str(dataset_path))["train"]

# ================= Load the model and template =================
print(f"Loading model from {model_id_or_path} ...", flush=True)
# Full fine-tuning loads the model directly without calling get_peft_model.
model, processor = get_model_processor(model_id_or_path)

# Required for gradient checkpointing.
model.enable_input_require_grads()

print("Initializing template ...", flush=True)
# Use the standard model template without modifying its internal parameters.
template = get_template(processor, max_length=max_length)
print(f'agent_template: {template._agent_template}')
template.set_mode('train')
if template.use_model:
    template.model = model

# ================= Build the lazy dataset =================
# Use the standard Swift path through template.encode. Swift reads the
# per-message "loss" fields and constructs the corresponding loss mask.
train_dataset = LazyLLMDataset(processed_dataset, template.encode, random_state=42)

# ================= Validate data and template encoding =================
print("=== Inspecting a sample after template encoding ===", flush=True)
while True:
    sample_data = random.choice(processed_dataset)
    # Encode a sample with the desired conversation length.
    if len(sample_data["messages"]) == 10:  # Other observed lengths include 3, 4, 6, 7, and 9.
        print(1)
    else:
        continue
    encoded_sample = template.encode(sample_data)

    input_ids = encoded_sample.get('input_ids', [])
    labels = encoded_sample.get('labels', [])

    # Decode the complete input seen by the model.
    full_text = processor.decode(input_ids)

    print("\n" + "="*30 + " [Complete model input (decoded input IDs)] " + "="*30, flush=True)
    print(full_text, flush=True)

    # Decode only the supervised tokens that contribute to the loss.
    valid_label_ids = [label for label in labels if label != -100]
    loss_text = processor.decode(valid_label_ids)

    print("\n" + "="*30 + " [Supervised text (decoded labels)] " + "="*30, flush=True)
    print(loss_text, flush=True)
    print("="*80 + "\n", flush=True)
    break
