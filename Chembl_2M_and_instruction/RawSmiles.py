import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from huggingface_hub import login

MODEL_ID = "allenai/OLMo-7B-hf"
NEW_ADAPTER_NAME = "Codemaster67/ChemOlmo-7b"

TOKEN = "HUGGING FACE TOKEN"

login(token=TOKEN)

class OLMoQLoRA(pl.LightningModule):
    def __init__(self, model_id, adapter_name):
        super().__init__()
        self.save_hyperparameters()
        self.model_id = model_id
        self.adapter_name = adapter_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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

        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(model, peft_config)
        
        if self.trainer.is_global_zero:
            self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-4)

        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(0.10 * total_steps)
        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


class ZINCDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = (
            load_dataset("antoinebcx/smiles-molecules-chembl", split="train", streaming=True, trust_remote_code=True)
            .take(100_000)
        )

    def train_dataloader(self):
        def collate_fn(batch):
            texts = [item['smiles'] for item in batch]

            encodings = self.tokenizer(
                texts,
                truncation=True,
                max_length=256,
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

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=16
        )

if __name__ == "__main__":
    pl_model = OLMoQLoRA(MODEL_ID, NEW_ADAPTER_NAME)

    dm = ZINCDataModule(pl_model.tokenizer, batch_size=32)

    trainer = pl.Trainer(
        accelerator="gpu", 
        precision="bf16-mixed",       
        max_epochs=1,
        log_every_n_steps=16,
        enable_checkpointing=False,
        gradient_clip_val=1
    )

    print("Starting Training on first 10k samples...")
    trainer.fit(pl_model, datamodule=dm)

    # ---------------------------------------------------------
    # MERGE AND PUSH LOGIC
    # ---------------------------------------------------------
    if trainer.is_global_zero:
        print("Training complete. Saving adapter to temporary local storage...")
        
        pl_model.model.save_pretrained("./temp_adapter")
        pl_model.tokenizer.save_pretrained("./temp_adapter")
        
        # Free up memory (VRAM) to allow loading the full base model for merging
        print("Freeing VRAM for merge process...")
        del pl_model
        del trainer
        del dm
        gc.collect()
        torch.cuda.empty_cache()

        print("Loading Base Model in FP16 for merging...")
        # Load base model in float16 (required for merging, cannot merge into 4bit)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        print("Loading Adapter and Merging...")
        model_to_merge = PeftModel.from_pretrained(base_model, "./temp_adapter")
        model_to_merge = model_to_merge.merge_and_unload()
        
        print(f"Pushing FULL MERGED model to Hub: {NEW_ADAPTER_NAME}")
        model_to_merge.push_to_hub(NEW_ADAPTER_NAME)
        
        # Push tokenizer as well
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.push_to_hub(NEW_ADAPTER_NAME)
        
        print("Done! Full merged model pushed.")
