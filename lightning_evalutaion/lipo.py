import torch
import pytorch_lightning as pl
import deepchem as dc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from deepchem.molnet import load_lipo 
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import mean_squared_error
import re
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

pl.seed_everything(42, workers=True)

class OlmoDataset(Dataset):
    def __init__(self, mode="Train", max_length=350):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Codemaster67/ChemOlmo-7b",
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        tasks, datasets, transformers = load_lipo(featurizer="raw", splitter='scaffold')
        train, valid, test = datasets

        self.mode = mode.lower()
        if self.mode == "train":
            self.data = train
        elif self.mode == "test":
            self.data = test

        elif self.mode == "valid": 
            self.data = valid
        
        self.max_length = max_length
        self.samples = []
        self._filldataset()

    def _filldataset(self):
        for i in range(len(self.data)):
            smiles = self.data.ids[i]
            labels = self.data.y[i]   
            weights = self.data.w[i] 

            for task_idx, label in enumerate(labels):
                if weights[task_idx] > 0:
                    self.samples.append(self._create_prompt(smiles, label))
            
        print(f"[{self.mode.upper()}] Number of samples: {len(self.samples)}")

    def _create_prompt(self, smiles, label):
        eos_token = self.tokenizer.eos_token
        answer = f"{label:.5f}"

        full_prompt = (
            "### Instruction:\n"
            "Predict the Lipophilicity (LogD) "
            f"for the following molecule:\n{smiles}\n\n"
            "### Response:\n"
            f"{answer}{eos_token}"
        )
        return full_prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        separator = "### Response:\n"
        parts = text.split(separator)

        if len(parts) >= 2:
            prompt_text = parts[0] + separator
            prompt_encodings = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            prompt_len = prompt_encodings["input_ids"].shape[1]

            if prompt_len < len(labels):
                labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class OLMO_QLoRA(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Codemaster67/ChemOlmo-7b",
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM 
        )
        
    def configure_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "Codemaster67/ChemOlmo-7b",
            quantization_config=self.bnb_config,
            trust_remote_code=True,
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
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
        self.log("Train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
            return loss

    def on_train_end(self):
            if self.trainer.is_global_zero:
                print("\nStarting test set evaluation (RMSE) after training...")

                test_dataset = OlmoDataset(mode="test", max_length=350)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                self.model.eval()

                preds = []
                trues = []
                

                true_values = []
                for i in range(len(test_dataset.data)):
                     if test_dataset.data.w[i][0] > 0:
                          true_values.append(test_dataset.data.y[i][0])
                
                print(f"Evaluating on {len(test_loader)} samples...")

                with torch.no_grad():
                    for i, batch in enumerate(test_loader):
                        batch = {k: v.to(self.device) for k, v in batch.items()}

                        input_ids = batch["input_ids"]
                        labels = batch["labels"]
                        attention_mask = batch["attention_mask"]

                        response_mask = (labels != -100)
                        if response_mask.sum() == 0:
                             continue
                             
                        answer_start_index = response_mask.int().argmax(dim=1).item()

                        if answer_start_index > 0:
                            prompt_ids = input_ids[:, :answer_start_index]
                            prompt_mask = attention_mask[:, :answer_start_index]
                        else:
                            prompt_ids = input_ids
                            prompt_mask = attention_mask

                        outputs = self.model.generate(
                            input_ids=prompt_ids,
                            attention_mask=prompt_mask,
                            max_new_tokens=10,
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=False
                        )

                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        try:
                            if "### Response:" in generated_text:
                                response_part = generated_text.split("### Response:")[-1].strip()
                            else:
                                response_part = generated_text.strip()

                            match = re.search(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)", response_part)
                            if match:
                                val = float(match.group())
                            else:
                                val = 0.0
                        except Exception:
                            val = 0.0

                        preds.append(val)
                        trues.append(true_values[i])

                        if i % 50 == 0:
                            print(f"Sample {i}: True={true_values[i]:.5f}, Pred={val:.5f}")

                preds = np.array(preds)
                trues = np.array(trues)

                rmse = np.sqrt(mean_squared_error(trues, preds))

                print("\n=== Test Set Metrics ===")
                print(f"RMSE: {rmse:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)

        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(0.15*total_steps)
        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.01,
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

if __name__ == "__main__":
    dataset = OlmoDataset(mode="train")
    valid_dataset = OlmoDataset(mode="valid")
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=3,     
        mode="min",     
        verbose=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=15,          
        precision="bf16-mixed",
        enable_checkpointing=False, 
        gradient_clip_val=1,
        callbacks=[early_stop_callback] 
    )

    model = OLMO_QLoRA()

    trainer.fit(model, train_loader, valid_loader)
