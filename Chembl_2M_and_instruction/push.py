import torch
from unsloth import FastLanguageModel
from huggingface_hub import login
import gc

# ====================== CONFIG ======================
# Use the checkpoint where loss was most stable (Step 1500 or 2000)
CHECKPOINT_PATH = "/home/saurav/Downloads/unsloth_checkpoints_checkpoint-20000.zip" 
NEW_REPO_NAME   = "Codemaster67/ChemOlmo2-7b"
TOKEN           = "NOne"

login(token=TOKEN)

# ====================== LOAD & PREP ======================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CHECKPOINT_PATH,
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)

# Crucial: Preps the model for the merge operation
model = FastLanguageModel.for_inference(model)

# ====================== MERGE + PUSH ======================
# Using 16bit (Float16) ensures it works with standard MoleculeNet benchmark scripts
model.save_pretrained_merged(
    NEW_REPO_NAME,
    tokenizer,
    save_method = "merged_16bit", 
    push_to_hub = True,
)

# Push tokenizer separately to ensure chat templates/EOS tokens are preserved
tokenizer.push_to_hub(NEW_REPO_NAME)

print(f"🚀 Success! View your model at: https://huggingface.co/{NEW_REPO_NAME}")

gc.collect()
torch.cuda.empty_cache()
