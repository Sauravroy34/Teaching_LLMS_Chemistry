import torch
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import gc
import random
from rdkit import Chem # Ensure rdkit is installed: pip install rdkit
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from huggingface_hub import login

# ========================= CONFIG =========================
MODEL_ID = "allenai/OLMo-7B-hf"
TOKENIZER_ID = "ChemFM/ChemFM-3B"

NEW_ADAPTER_NAME = "Codemaster67/ChemOlmoAtomTok-7b"
TOKEN = "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"

login(token=TOKEN)
random.seed(42)

def enumerate_smiles(smi: str, n: int = 10):
    """Generates randomized SMILES variants for data augmentation"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    return [Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(n)]

class OLMoQLoRA(pl.LightningModule):
    def __init__(self, model_id, tokenizer_id):
        super().__init__()
        self.save_hyperparameters()
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def configure_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            trust_remote_code=True
        )

        model.resize_token_embeddings(len(self.tokenizer))

        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            modules_to_save=["embed_tokens", "lm_head"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(model, peft_config)
        
        if self.trainer.is_global_zero:
            self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"] 
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5, weight_decay=1e-4)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.10 * total_steps)

        scheduler_warmup = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ====================== AUGMENTED DATA MODULE ======================
class ZINCDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=32):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load stream
        raw_dataset = load_dataset("antoinebcx/smiles-molecules-chembl", split="train", streaming=True)
        
        def augment_example(example):
            aug_list = enumerate_smiles(example["smiles"], n=10)
            return {"smiles": random.choice(aug_list)}

        self.dataset = raw_dataset.map(augment_example)

    def train_dataloader(self):
        def collate_fn(batch):
            texts = [item['smiles'] for item in batch]
            encodings = self.tokenizer(
                texts,
                truncation=True,
                max_length=512, 
                padding="max_length",
                return_tensors="pt"
            )
            labels = encodings["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels
            }

        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=4)

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    pl_model = OLMoQLoRA(MODEL_ID, TOKENIZER_ID)
    dm = ZINCDataModule(pl_model.tokenizer, batch_size=16) 
    trainer = pl.Trainer(
        accelerator="gpu", 
        precision="bf16-mixed",        
        max_epochs=1,
        max_steps = 12000,
        log_every_n_steps=10,
        enable_checkpointing=False,
        gradient_clip_val=1.0
    )

    print("Starting pre-training with Atom-Level Tokenizer + SMILES Augmentation...")
    trainer.fit(pl_model, datamodule=dm)

    # MERGE LOGIC
    if trainer.is_global_zero:
        print("Saving and Merging...")
        pl_model.model.save_pretrained("./temp_adapter")
        final_vocab_size = len(pl_model.tokenizer)

        del pl_model, trainer, dm
        gc.collect()
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        base_model.resize_token_embeddings(final_vocab_size)
        
        model_to_merge = PeftModel.from_pretrained(base_model, "./temp_adapter")
        model_to_merge = model_to_merge.merge_and_unload()
        
        model_to_merge.push_to_hub(NEW_ADAPTER_NAME)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
        tokenizer.push_to_hub(NEW_ADAPTER_NAME)
        print("✅ Merged model with augmented SMILES training is live!")
