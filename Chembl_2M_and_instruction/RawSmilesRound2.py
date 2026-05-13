import torch
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, interleave_datasets
from huggingface_hub import login
import gc

MODEL_ID = "Codemaster67/ChemOlmo-7b"
NEW_ADAPTER_NAME = "Codemaster67/ChemOlmo-7b-2nd"
TOKEN = "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"

login(token=TOKEN)

# ====================== Unsloth FastLanguageModel ======================
print("Loading Unsloth model (2x faster + 70% less VRAM)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=512,
    dtype=None,                   
    load_in_4bit=True,
    trust_remote_code=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",   # Unsloth's ultra-fast version
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("Trainable parameters:")
model.print_trainable_parameters()

# ====================== 5 Million Samples (balanced) ======================
print("Creating 5M interleaved dataset (PubChem + ZINC)...")
pubchem = load_dataset("jablonkagroup/pubchem-smiles-molecular-formula", split="train", streaming=True).take(1_000_000)
zinc    = load_dataset("haydn-jones/ZINC20", split="train", streaming=True).take(1_000_000)

dataset = interleave_datasets(
    [pubchem, zinc],
    probabilities=[0.5, 0.5],
    stopping_strategy="all_exhausted",
    seed=42,
).shuffle(buffer_size=10_000)

# Add "text" field for SFTTrainer (SMILES + EOS)
def format_example(example):
    smiles = example.get("smiles") or example.get("SMILES") or ""
    return {"text": str(smiles).strip() + tokenizer.eos_token}

dataset = dataset.map(format_example, batched=False)

# ====================== SFTTrainer with Sequence Packing ======================
training_args = TrainingArguments(
    output_dir="./unsloth_checkpoints",
    per_device_train_batch_size=64,          
    gradient_accumulation_steps=2,         
    warmup_steps=500,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),           
    logging_steps=10,
    optim="adamw_8bit",
    report_to="none",
    max_steps=15625,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",              # <-- Correctly placed here
    max_seq_length=512,                     # <-- Correctly placed here
    dataset_num_proc=4,
    packing=True,                           # <-- Correctly placed here
    args=training_args,
)
print("Starting continued pre-training on 5M SMILES with Unsloth + packing...")
trainer.train()

# ====================== Merge & Push (Unsloth way - fastest) ======================
if True:  # change to trainer.is_global_zero if multi-GPU
    print("Merging LoRA into full model...")
    model = FastLanguageModel.for_inference(model)  # optional
    
    model.save_pretrained_merged(
        NEW_ADAPTER_NAME,
        tokenizer,
        save_method="merged_16bit",   # full 16-bit model
        push_to_hub=True,
        private=False,
    )
    
    print(f"✅ Full merged model pushed to: https://huggingface.co/{NEW_ADAPTER_NAME}")
    tokenizer.push_to_hub(NEW_ADAPTER_NAME)

gc.collect()
torch.cuda.empty_cache()