"""
Chemistry Causal LM Dataset Preparation
Prepares and pushes to Codemaster67/Causal_lm_chemistry

"""

import random
import logging
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained(
"Codemaster67/Olmo-7b-spe",
trust_remote_code=True,
) 




SOM = "<|start_of_smiles|>"
EOM = "<|end_of_smiles|>"

# This is done to randomize the order smiles and text position 
TEMPLATES = [
    "{som}{smiles}{eom}\n{description}",

    "The chemical structure {som}{smiles}{eom} can be described as follows: {description}",

    "{description} This compound is represented by the structure {som}{smiles}{eom}.",

    "Regarding {som}{smiles}{eom}, {description}",

    "Based on its properties, {description} Structurally, we map this to {som}{smiles}{eom}.",
]



def prepare_unichem(n: int = 300_000):

    # The dataset is large; stream to avoid OOM
    ds = load_dataset(
        "bisectgroup/UniChem",
        split="train",
        streaming=True,
    )

    records = []
    seen = set()

    for row in ds:
        smiles = row.get("smiles") or row.get("standardised_smiles") or row.get("SMILES")
        if not smiles:
            continue
        smiles = smiles.strip()
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        records.append({"text": f"{SOM}{smiles}{EOM}", "source": "unichem"})
        if len(records) >= n:
            break

    return Dataset.from_list(records)



def prepare_chemrxiv(n: int = 30_000):

    ds = load_dataset(
        "BASF-AI/ChemRxiv-Papers",
        split="train",
        streaming=True,
    )

    records = []
    for row in ds:
        title    = (row.get("title")    or "").strip()
        abstract = (row.get("abstract") or "").strip()

        if not abstract:          # skip rows with no abstract
            continue

        text = f"{title}\n\n{abstract}" if title else abstract
        records.append({"text": text, "source": "chemrxiv"})

        if len(records) >= n:
            break

    return Dataset.from_list(records)


def prepare_chebi20(n: int = 100_000) -> Dataset:

    raw_1 = load_dataset("liupf/ChEBI-20-MM")
    raw_2 = load_dataset("liuganghuggingface/moltextnet")

    # Helper function to extract and normalize columns across splits
    def extract_pairs(raw_ds, smiles_col):
        for split in raw_ds.keys():
            for row in raw_ds[split]:
                smiles = (row.get(smiles_col) or "").strip()
                description = (row.get("description") or row.get("text") or "").strip()
                if smiles and description:
                    yield smiles, description

    # Chain both datasets together, mapping to the correct column names
    import itertools
    all_pairs = list(itertools.chain(
        extract_pairs(raw_1, smiles_col="SMILES"),
        extract_pairs(raw_2, smiles_col="canonical_smiles")
    ))

    random.shuffle(all_pairs)

    records = []
    for smiles, description in all_pairs:
        template = random.choice(TEMPLATES)
        text = template.format(
            som=SOM,
            eom=EOM,
            smiles=smiles,
            description=description,
        )
        
        # Tokenize and filter out sequences longer than 512 tokens
        tokens = tokenizer.encode(text)
        if len(tokens) > 512:
            continue

        records.append({"text": text, "source": "chebi20_moltextnet"})

        if len(records) >= n:
            break

    return Dataset.from_list(records)


def prepare_pubmed(n: int = 30_000):
    ds = load_dataset(
        "casinca/PUBMED_title_abstracts_2019_baseline",
        split="train"
    )

    records = []
    for row in ds:
        Text    = (row.get("text") or "").strip()
        if not Text:
            continue
        records.append({"text": Text, "source": "pubmed"})

        if len(records) >= n:
            break

    return Dataset.from_list(records)


def build_and_push(hub_repo = "Codemaster67/Causal_lm_chemistry"):
    random.seed(42)

    unichem_ds  = prepare_unichem(1_000_000)
    chemrxiv_ds = prepare_chemrxiv(30_000)
    chebi_ds    = prepare_chebi20(500_000)
    pubmed_ds   = prepare_pubmed(20_000)

    combined = concatenate_datasets([unichem_ds, chemrxiv_ds, chebi_ds, pubmed_ds])
    combined = combined.shuffle(seed=42)


    split = combined.train_test_split(test_size=0.05, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "test":  split["test"],
    })


    # ── push ──────────────────────────────────────────────────────────────────
    dataset_dict.push_to_hub(
        hub_repo,
        private=False,          
    )

    return dataset_dict


if __name__ == "__main__":
    ds = build_and_push()

    print("------ dataset pushed to hub ----------")
    for source in ("unichem", "chemrxiv", "chebi20", "pubmed"):
        sample = ds["train"].filter(lambda x: x["source"] == source).select(range(1))[0]
        print(f"\n{'─'*60}")
        print(f"[{source}]  {sample['text'][:300]}")