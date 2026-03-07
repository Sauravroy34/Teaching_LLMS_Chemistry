import torch
import pytorch_lightning as pl
import re
import bitsandbytes as bnb 
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class ChemCoTHFDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_length=60000, limit=10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Streaming ChemCoT for Expert Reasoning with max_length={max_length}...")
        ds = load_dataset("OpenMol/ChemCoTDataset", split="train", streaming=True)
        
        self.samples = []
        for i, entry in enumerate(ds):
            if i >= limit: break
            
            match = re.search(r'["\']output["\']:\s*["\']([^"\']+)["\']', str(entry['struct_cot']))
            
            if match:
                final_answer = match.group(1)
                
                trajectory = (
                    "### Instruction:\n"
                    "[Role] You are an expert chemist specializing in molecular property prediction.\n"
                    "[Task] Provide a systematic chemical reasoning process covering molecular structure analysis,"
                    "chemical principles, and comparative analysis. Ensure you perform enough thinking before"
                    "providing the final answer.\n"
                    f"Question: {entry['query']}\n\n"
                    f"### Response:\n<think>\n{entry['raw_cot']}\n</think>\n"
                    f"<answer>{final_answer}</answer>"
                )
                self.samples.append(trajectory)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx] + self.tokenizer.eos_token
        encodings = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        separator = "### Response:\n"
        if separator in text:
            prompt_part = text.split(separator)[0] + separator
            prompt_len = len(self.tokenizer.encode(prompt_part, add_special_tokens=False))
            labels[:prompt_len] = -100
            
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": encodings["attention_mask"].squeeze(0), "labels": labels}

# 3. QLoRA Training Module
class OLMoThinkTrainer(pl.LightningModule):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def configure_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            quantization_config=self.bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
        peft_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"],
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, peft_config)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.03 * total_steps)
        
        sched1 = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
        sched2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": SequentialLR(optimizer, [sched1, sched2], [warmup_steps]), "interval": "step"}}

if __name__ == "__main__":
    MODEL_ID = "allenai/Olmo-3-7B-Think"
    HUB_NAME = "olmo-3-7b-think_chemcot"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = ChemCoTHFDataset(tokenizer, max_length=50000, limit=10000)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1, 
        max_epochs=1, 
        precision="bf16-mixed",
        accumulate_grad_batches=16, 
        gradient_clip_val=1.0
    )

    model_module = OLMoThinkTrainer(MODEL_ID)
    trainer.fit(model_module, loader)

    if trainer.is_global_zero:
        print("Finalizing Training: Merging adapters into base weights...")
        adapter_path = "./temp_adapter"
        model_module.model.save_pretrained(adapter_path)
        
        del model_module.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()
        
        print(f"Pushing full expert model to Hub: {HUB_NAME}")
        merged_model.push_to_hub(HUB_NAME)
        tokenizer.push_to_hub(HUB_NAME)

