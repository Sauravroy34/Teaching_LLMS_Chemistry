import torch
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import gc, random
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from huggingface_hub import login
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


MODEL_ID = "Codemaster67/ChemOlmo-7b"
NEW_ADAPTER_NAME = "Codemaster67/ChemOlmo-7b-Instruct"
TOKEN = "HUGGING FACE TOKEN"

login(token=TOKEN)

PROMPT_IUPAC = """### Instruction:
Convert the following SMILES string to its IUPAC name. Answer with only the IUPAC name.

{smiles}

### Response:
"""

PROMPT_REACTION = """### Instruction:
Given the reactants and reagents in SMILES format, predict the major product SMILES.

{reactants_reagents}

### Response:
"""

PROMPT_PROPERTY = """### Instruction:
Given the SMILES string, predict its molecular weight (g/mol), TPSA (Å²), and QED score.
Output exactly in this format: MW: X.XXXX, TPSA: X.XXXX, QED: X.XXXX

{smiles}

### Response:
"""

PROMPT_GENERAL = """### Instruction:
{instruction}

### Response:
"""

# =============================================================================

class OLMoQLoRA(pl.LightningModule):
    def __init__(self, model_id, adapter_name):
        super().__init__()
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
            r=32,
            lora_alpha=64,
            target_modules="all-linear",
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(model, peft_config)
        # Safe trainer check
        if getattr(self, "trainer", None) is not None and self.trainer.is_global_zero:
            self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5, weight_decay=1e-4)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.10 * total_steps)
        scheduler_warmup = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

class MixedChemDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=32):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        iupac = load_dataset("hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC", split="train", streaming=True).take(10_000)
        reactions = load_dataset("pingzhili/uspto-50k", split="train", streaming=True).take(10_000)
        properties = load_dataset("hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC", split="train", streaming=True).take(10_000)
        general = load_dataset("AlgorithmicResearchGroup/arxiv-cs-ml-instruct-tune-50k", split="train").take(15000)
        replay = load_dataset("jablonkagroup/pubchem-smiles-molecular-formula", split="train", streaming=True).take(35000)
        zinc = load_dataset("haydn-jones/ZINC20", split="train", streaming=True).take(35000)

        data_list = []
        
        # Add task tags
        for ex in list(iupac): ex['task'] = 'iupac'; data_list.append(ex)
        for ex in list(reactions): ex['task'] = 'reaction'; data_list.append(ex)
        for ex in list(properties): ex['task'] = 'property'; data_list.append(ex)
        for ex in list(general): ex['task'] = 'general'; data_list.append(ex)
        for ex in list(replay): ex['task'] = 'replay'; data_list.append(ex)
        for ex in list(zinc): ex['task'] = 'zinc'; data_list.append(ex)

        random.shuffle(data_list)
        self.dataset = data_list

    def train_dataloader(self):
        def collate_fn(batch):
            input_ids_list = []
            attention_mask_list = []
            labels_list = []

            for item in batch:
                task = item['task']
                
                if task == 'iupac':
                    smiles = item.get('SMILES', '')
                    iupac = item.get('iupac') or item.get('iupac_name', '')
                    prompt = PROMPT_IUPAC.format(smiles=smiles)
                    target = iupac
                    
                elif task == 'reaction':
                    rxn = item['rxn_smiles']
                    prod = item['prod_smiles']
                    reactants = rxn.split('>>')[0]
                    prompt = PROMPT_REACTION.format(reactants_reagents=reactants)
                    target = prod
                    
                elif task == 'property':
                    smiles = item.get('SMILES', '')
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        mw = round(Descriptors.MolWt(mol), 4)
                        tpsa = round(Descriptors.TPSA(mol), 4)
                        qed = round(QED.qed(mol), 4)
                        target = f"MW: {mw}, TPSA: {tpsa}, QED: {qed}"
                    else:
                        target = "MW: 0.0000, TPSA: 0.0000, QED: 0.0000"
                    prompt = PROMPT_PROPERTY.format(smiles=smiles)
                    
                elif task == 'general':
                    prompt = PROMPT_GENERAL.format(instruction=item.get('question', ''))
                    target = item.get('answer', '')
                    
                else:  # replay AND zinc (Pre-training)
                    text = item.get('smiles', '') or item.get('text', '') or item.get('SMILES', '')
                    
                    full_text = text + self.tokenizer.eos_token
                    encodings = self.tokenizer(full_text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
                    input_ids = encodings["input_ids"][0]
                    attention_mask = encodings["attention_mask"][0]
                    labels = input_ids.clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                    input_ids_list.append(input_ids)
                    attention_mask_list.append(attention_mask)
                    labels_list.append(labels)
                    continue

                # Common path for prompted tasks (Instruction Tuning)
                full_text = prompt + target + self.tokenizer.eos_token
                full_enc = self.tokenizer(full_text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
                prompt_enc = self.tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")

                input_ids = full_enc["input_ids"][0]
                attention_mask = full_enc["attention_mask"][0]
                labels = input_ids.clone()
                prompt_len = prompt_enc["input_ids"].shape[1]
                labels[:prompt_len] = -100
                labels[input_ids == self.tokenizer.pad_token_id] = -100

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                labels_list.append(labels)

            return {
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "labels": torch.stack(labels_list)
            }

        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=16, shuffle=True)

# ========================= MAIN =========================
if __name__ == "__main__":

    pl_model = OLMoQLoRA(MODEL_ID, NEW_ADAPTER_NAME)
    dm = MixedChemDataModule(pl_model.tokenizer, batch_size=16)
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="bf16-mixed",
        max_epochs=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        gradient_clip_val=1.0
    )

    print("Starting Mixed Instruction Tuning (150k samples - including ZINC20)...")
    trainer.fit(pl_model, datamodule=dm)

    # Merge & Push 
    if getattr(trainer, "is_global_zero", True): 
        print("Training complete. Saving adapter...")
        pl_model.model.save_pretrained("./temp_adapter")
        pl_model.tokenizer.save_pretrained("./temp_adapter")
        
        del pl_model; del trainer; del dm
        gc.collect(); torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, return_dict=True, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True
        )
        merged_model = PeftModel.from_pretrained(base_model, "./temp_adapter").merge_and_unload()
        print(f"Pushing final model → {NEW_ADAPTER_NAME}")
        merged_model.push_to_hub(NEW_ADAPTER_NAME)
        AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True).push_to_hub(NEW_ADAPTER_NAME)
        print("✅ All done!")