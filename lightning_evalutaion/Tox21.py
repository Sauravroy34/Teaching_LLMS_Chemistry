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
from pytorch_lightning.callbacks import ModelCheckpoint 
from deepchem.molnet import load_tox21
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import accuracy_score, roc_auc_score
import re
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class OlmoDataset(Dataset):
    def __init__(self, mode="Train", max_length=600):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Codemaster67/ChemOlmo-7b",
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token


        tasks, datasets, transformers = load_tox21(featurizer="raw", splitter='scaffold')
        train, valid, test = datasets
        
        self.task_names = tasks # List of 12 assay names (e.g. NR-AhR)

        self.mode = mode.lower()
        if self.mode == "train":
            self.data = train
        elif self.mode == "valid":
            self.data = valid
        elif self.mode == "test":
            self.data = test

        self.max_length = max_length
        self.samples = []
        self._filldataset()

    def _filldataset(self):

        for i in range(len(self.data)):
            smiles = self.data.ids[i]
            labels = self.data.y[i]   
            weights = self.data.w[i]  #it has some missing data too

            for task_idx, label in enumerate(labels):
                #Only train on valid data (weight > 0)
                if weights[task_idx] > 0:
                    task_name = self.task_names[task_idx]
                    self.samples.append(self._create_prompt(smiles, task_name, label))
                    
        print(f"[{self.mode.upper()}] Number of samples: {len(self.samples)}")

    def _create_prompt(self, smiles, task_name, label):
        eos_token = self.tokenizer.eos_token
        
        answer = "Yes" if label == 1.0 else "No"

        full_prompt = (
            "### Instruction:\n"
            f"Is the following molecule toxic in the {task_name} assay?\n"
            f"{smiles}\n\n"
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
        
        if (labels != -100).sum() == 0:
            labels[-1] = input_ids[-1]

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
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def on_train_end(self):
            if self.trainer.is_global_zero:
                print("\nStarting test set evaluation (Accuracy) after training...")

                test_dataset = OlmoDataset(mode="test", max_length=600)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

                self.model.eval()

                preds = []
                trues = []

                print(f"Evaluating on {len(test_loader)} samples...")

                with torch.no_grad():
                    for i, batch in enumerate(test_loader):

                        batch = {k: v.to(self.device) for k, v in batch.items()}

                        input_ids = batch["input_ids"]
                        labels = batch["labels"]
                        attention_mask = batch["attention_mask"]

                        response_mask = (labels != -100)
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
                            max_new_tokens=5, 
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=False
                        )

                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        try:
                            if "### Response:" in generated_text:
                                response_part = generated_text.split("### Response:")[-1].strip()
                            else:
                                response_part = generated_text.strip()

                            # Simple string matching
                            if "Yes" in response_part:
                                pred_label = "Yes"
                            elif "No" in response_part:
                                pred_label = "No"
                            else:
                                pred_label = "Invalid"
                            
                            
                            truth_ids = labels[0][labels[0] != -100]
                            truth_text = self.tokenizer.decode(truth_ids, skip_special_tokens=True).strip()
                            
                            preds.append(pred_label)
                            trues.append(truth_text)

                        except Exception as e:
                            print(f"Error parsing prediction: {e}")

                        if i % 10 == 0:
                            print(f"Sample {i}: True={trues[-1]}, Pred={preds[-1]}")

                acc = accuracy_score(trues, preds)
                
                print("\n=== Test Set Metrics ===")
                print(f"Accuracy: {acc:.4f}")
                
                roc = roc_auc_score(trues, preds)
                print(f"ROC: {roc:.4f}")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4,weight_decay= 1e-4)

        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(0.15*total_steps)
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


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    dataset = OlmoDataset()
    valid_dataset = OlmoDataset(mode="valid")

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,      
        mode="min",
        verbose=True
    )
    trainer = pl.Trainer(
                accelerator="gpu",
                max_epochs=1, 
                precision="bf16-mixed",
                enable_checkpointing=False,
                gradient_clip_val=1,
                log_every_n_steps=10,
                callbacks=[early_stop_callback]
                
            )
    
    model = OLMO_QLoRA()
    
    trainer.fit(model, train_loader, valid_loader)
    
