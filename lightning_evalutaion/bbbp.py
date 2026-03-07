import torch
import pytorch_lightning as pl
import deepchem as dc
import numpy as np
import pandas as pd
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
from deepchem.molnet import load_bbbp
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.functional import softmax
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class OlmoDataset(Dataset):
    def __init__(self, mode="Train", max_length=350): # Bumped to 350 to prevent truncation NaNs
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Codemaster67/ChemOlmo-7b",
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        tasks, datasets, transformers = load_bbbp(featurizer="raw", splitter='scaffold')
        train, valid, test = datasets
        
        self.task_names = tasks 

        self.mode = mode.lower()
        if self.mode == "train":
            self.data = train
        elif self.mode == "valid": # Added valid mode explicitly
            self.data = valid
        elif self.mode == "test":
            self.data = test
        else:
            self.data = train 

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
                    task_name = "blood-brain barrier penetration"
                    self.samples.append(self._create_prompt(smiles, task_name, label))
                     
        print(f"[{self.mode.upper()}] Number of samples: {len(self.samples)}")

    def _create_prompt(self, smiles, task_name, label):
        eos_token = self.tokenizer.eos_token
        answer = "Yes" if label == 1.0 else "No"
        
        full_prompt = (
            "### Instruction:\n"
            f"Is the following molecule capable of {task_name}?\n"
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
        
        # Safeguard: If everything is masked, unmask the last token to prevent NaN loss
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

    # Added validation_step to allow EarlyStopping to function
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
            print("\nStarting test set evaluation (Accuracy & ROC-AUC)...")
            
            test_dataset = OlmoDataset(mode="test", max_length=350)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
            
            self.model.eval()
            
            y_true = []
            y_probs = [] 
            y_pred = []  

            print(f"Evaluating on {len(test_loader)} samples using Logits...")

            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]
                    attention_mask = batch["attention_mask"]

                    response_mask = (labels != -100)
                    if response_mask.sum() == 0: continue
                    
                    answer_start_index = response_mask.int().argmax(dim=1).item()
                    
                    if answer_start_index > 0:
                        prompt_ids = input_ids[:, :answer_start_index]
                        prompt_mask = attention_mask[:, :answer_start_index]
                    else:
                        continue 

                    outputs = self.model(input_ids=prompt_ids, attention_mask=prompt_mask)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    yes_logit = next_token_logits[:, yes_token_id]
                    no_logit = next_token_logits[:, no_token_id]
                    
                    relevant_logits = torch.stack([no_logit, yes_logit], dim=1)
                    probs = softmax(relevant_logits, dim=1)
                    
                    prob_yes = probs[:, 1].item()
                    prediction = 1 if prob_yes > 0.5 else 0
                    
                    truth_id = labels[0][answer_start_index].item()
                    if truth_id == yes_token_id:
                        true_label = 1
                    elif truth_id == no_token_id:
                        true_label = 0
                    else:
                        continue 

                    y_true.append(true_label)
                    y_probs.append(prob_yes)
                    y_pred.append(prediction)

                    if i % 100 == 0:
                        print(f"Sample {i}: True={true_label}, Prob(Yes)={prob_yes:.4f}")

            acc = accuracy_score(y_true, y_pred)
            roc = roc_auc_score(y_true, y_probs)

            print("\n=== Test Set Metrics ===")
            print(f"Accuracy: {acc:.4f}")
            print(f"ROC-AUC:  {roc:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.15*total_steps)
        scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

if __name__ == "__main__":
    pl.seed_everything(42)

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
        max_epochs=3,  
        precision="bf16-mixed", 
        gradient_clip_val=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        callbacks=[early_stop_callback] 
    )
   
    model = OLMO_QLoRA()
   
    trainer.fit(model, train_loader, val_dataloaders=valid_loader)
