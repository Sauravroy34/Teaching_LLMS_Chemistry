import random
from datasets import load_dataset, interleave_datasets
from rdkit import Chem

from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer

# ========================= CONFIG =========================
MODEL_NAME = "allenai/OLMo-7B"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4                          
GRAD_ACCUM = 8
LEARNING_RATE = 4e-4
NUM_EPOCHS = 1
RANK = 64
ALPHA = 128

OUTPUT_DIR = "./olmo-7b-chem-pretrain-qlora-r64"
HF_REPO_ID = "Codemaster67/ChemOLmo2-7b"   
random.seed(42)
# =========================================================

def enumerate_smiles(smi: str, n: int = 10):
    """10× randomized SMILES augmentation (ChemFM style)"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    return [Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(n)]

# ====================== LOAD DATA FROM HF ======================
print("Loading datasets from Hugging Face (streaming)...")

# ChEMBL (~1.94M molecules)
chembl = load_dataset("antoinebcx/smiles-molecules-chembl", split="train", streaming=True)
chembl = chembl.rename_column("smiles", "text")

# Large dataset: datamol-io/safe-gpt (~1.17B molecules from ZINC20 + UniChem)
large_ds = load_dataset("datamol-io/safe-gpt", split="train", streaming=True)
large_ds = large_ds.rename_column("smiles", "text")

# Sample ~176M from the large dataset to reach ~178M total
target_large = 176_000_000
large_sampled = large_ds.take(target_large)

# Interleave (mostly large dataset + small portion ChEMBL)
dataset = interleave_datasets([chembl, large_sampled], probabilities=[0.10, 0.90])

# On-the-fly 10× augmentation
def augment_example(example):
    aug_list = enumerate_smiles(example["text"], n=10)
    return {"text": random.choice(aug_list)}  

dataset = dataset.map(augment_example, batched=False)

print("Dataset ready with streaming + on-the-fly augmentation.")

# ====================== LOAD MODEL & APPLY QLoRA ======================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Resize if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    target_modules= "all_linear",
    lora_alpha=ALPHA,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)

# ====================== TRAINING ======================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_strategy="steps",
        save_steps=10000,
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
        packing=True,
        dataloader_num_workers=4,
    ),
)

print("Starting pre-training...")
trainer.train()

# ====================== MERGE & PUSH TO HF HUB ======================
print("Merging LoRA weights into base model...")

# Merge LoRA + base model
model = FastLanguageModel.for_inference(model)  # Optional, cleans up
merged_model = model.merge_and_unload()         # This merges the adapter

print(f"Pushing merged model to Hugging Face: {HF_REPO_ID}")

# Login (leave token empty - you will be prompted or use HF_TOKEN env var)
merged_model.push_to_hub(
    repo_id=HF_REPO_ID,
    token="",               # ← Leave empty as requested (you fill it)
    private=False,          # Change to True if you want private repo
    safe_serialization=True,
)

tokenizer.push_to_hub(
    repo_id=HF_REPO_ID,
    token="",
)

print(f"✅ Done! Merged model successfully pushed to https://huggingface.co/{HF_REPO_ID}")