"""
CLEANMOL Data Generator
========================
Generates structure-aware SMILES parsing tasks for pretraining.
Run this on CPU BEFORE renting GPU time.

Tasks:
  1. Functional group matching (binary)
  2. Ring counting (integer)
  3. Carbon chain length (integer)
  4. SMILES canonicalization (string)
  5. Fragment assembly (string)

Usage:
  python 01_generate_cleanmol_data.py --output_dir ./cleanmol_data --num_molecules 50000
  python 01_generate_cleanmol_data.py --output_dir ./cleanmol_data --num_molecules 50000 --push_to_hub
"""

import argparse
import json
import os
import random
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, BRICS
from rdkit.Chem import rdmolops

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Common functional groups (SMARTS)
# ---------------------------------------------------------------------------
FUNCTIONAL_GROUPS = {
    "hydroxyl": "[OX2H]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[CX3](=O)[OX2H0]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfonyl": "[SX4](=O)(=O)",
    "ether": "[OD2]([#6])[#6]",
    "thiol": "[SX2H]",
    "halide": "[F,Cl,Br,I]",
    "nitrile": "[CX2]#[NX1]",
    "aromatic_ring": "c1ccccc1",
    "phenol": "[OX2H]c1ccccc1",
}

# ---------------------------------------------------------------------------
# SMILES enumeration helpers
# ---------------------------------------------------------------------------
def randomize_smiles(mol, n=1):
    """Generate n random SMILES for a molecule."""
    results = set()
    for _ in range(n * 5):
        smi = Chem.MolToSmiles(mol, doRandom=True)
        results.add(smi)
        if len(results) >= n:
            break
    return list(results)


def get_longest_carbon_chain(mol):
    """Find the longest acyclic carbon chain (excluding ring atoms)."""
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)

    carbon_atoms = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetIdx() not in ring_atoms
    ]

    if len(carbon_atoms) < 2:
        return len(carbon_atoms)

    # Build adjacency among acyclic carbons
    adj = defaultdict(list)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in carbon_atoms and j in carbon_atoms:
            # carbon_atoms is a list, check membership
            if i not in ring_atoms and j not in ring_atoms:
                adj[i].append(j)
                adj[j].append(i)

    # BFS from every acyclic carbon to find longest path
    best = 1
    for start in carbon_atoms:
        visited = {start}
        queue = [(start, 1)]
        while queue:
            node, depth = queue.pop(0)
            best = max(best, depth)
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, depth + 1))
    return best


def fragment_molecule(mol):
    """
    Break a molecule into two BRICS fragments.
    Returns (frag1, frag2, original_canonical) or None if fragmentation fails.
    """
    try:
        frags = list(BRICS.BRICSDecompose(mol, minFragmentSize=3))
        if len(frags) >= 2:
            return frags[0], frags[1], Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------
def gen_functional_group_task(smi, mol):
    """Generate a functional group presence/absence question."""
    tasks = []
    # Pick 2 random groups: one present, one absent
    present = []
    absent = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            present.append(name)
        else:
            absent.append(name)

    # Generate a positive and negative example
    if present:
        grp = random.choice(present)
        tasks.append({
            "task": "functional_group",
            "input": (
                f"### Instruction:\n"
                f"Does the molecule contain a {grp} group?\n"
                f"{smi}\n\n"
                f"### Response:\n"
                f"Yes"
            ),
        })
    if absent:
        grp = random.choice(absent)
        tasks.append({
            "task": "functional_group",
            "input": (
                f"### Instruction:\n"
                f"Does the molecule contain a {grp} group?\n"
                f"{smi}\n\n"
                f"### Response:\n"
                f"No"
            ),
        })
    return tasks


def gen_ring_counting_task(smi, mol):
    """Generate a ring counting question."""
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]

    tasks = []
    for size in [5, 6]:
        count = sum(1 for s in ring_sizes if s == size)
        size_word = "five" if size == 5 else "six"
        tasks.append({
            "task": "ring_counting",
            "input": (
                f"### Instruction:\n"
                f"Count the {size_word}-membered rings in the following molecule.\n"
                f"{smi}\n\n"
                f"### Response:\n"
                f"{count}"
            ),
        })
    return tasks


def gen_chain_length_task(smi, mol):
    """Generate a carbon chain length question."""
    length = get_longest_carbon_chain(mol)
    return [{
        "task": "chain_length",
        "input": (
            f"### Instruction:\n"
            f"What is the length of the longest acyclic carbon chain in the molecule?\n"
            f"{smi}\n\n"
            f"### Response:\n"
            f"{length}"
        ),
    }]


def gen_canonicalization_task(smi, mol):
    """Generate a SMILES canonicalization question."""
    canonical = Chem.MolToSmiles(mol)
    randoms = randomize_smiles(mol, n=1)
    if not randoms or randoms[0] == canonical:
        # Try harder
        randoms = randomize_smiles(mol, n=3)
        randoms = [r for r in randoms if r != canonical]

    if not randoms:
        return []

    noncanonical = randoms[0]
    return [{
        "task": "canonicalization",
        "input": (
            f"### Instruction:\n"
            f"Canonicalize the following SMILES string.\n"
            f"{noncanonical}\n\n"
            f"### Response:\n"
            f"{canonical}"
        ),
    }]


def gen_fragment_assembly_task(smi, mol):
    """Generate a fragment assembly question."""
    result = fragment_molecule(mol)
    if result is None:
        return []
    frag1, frag2, original = result
    return [{
        "task": "fragment_assembly",
        "input": (
            f"### Instruction:\n"
            f"Assemble these two molecular fragments into a single molecule.\n"
            f"Fragment 1: {frag1}\n"
            f"Fragment 2: {frag2}\n\n"
            f"### Response:\n"
            f"{original}"
        ),
    }]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def load_molecules(num_molecules):
    smiles_list = []

    # Try ZINC250K first

    from datasets import load_dataset
    print("Loading ZINC250K from HuggingFace...")
    ds = load_dataset("yairschiff/zinc250k", split="train", streaming=True)
    for i, item in enumerate(ds):
        if i >= num_molecules:
            break
        smi = item.get("smiles", "")
        if smi:
            smiles_list.append(smi)
    print(f"Loaded {len(smiles_list)} molecules from ZINC250K")



def generate_all_tasks(smiles_list, max_per_task=50000):
    """Generate CLEANMOL tasks from SMILES list."""
    all_tasks = []
    task_counts = defaultdict(int)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Skip molecules that are too simple or too complex
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 5 or n_atoms > 80:
            continue

        canonical = Chem.MolToSmiles(mol)

        # Generate tasks for this molecule
        for gen_fn in [
            gen_functional_group_task,
            gen_ring_counting_task,
            gen_chain_length_task,
            gen_canonicalization_task,
            gen_fragment_assembly_task,
        ]:
            try:
                tasks = gen_fn(canonical, mol)
                for t in tasks:
                    if task_counts[t["task"]] < max_per_task:
                        all_tasks.append(t)
                        task_counts[t["task"]] += 1
            except Exception:
                continue

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(smiles_list)} molecules, "
                  f"generated {len(all_tasks)} tasks so far")
            for k, v in sorted(task_counts.items()):
                print(f"    {k}: {v}")

    return all_tasks


def push_dataset_to_hub(output_file, repo_name, token, test_size=0.05):
    """
    Push generated CLEANMOL data to HuggingFace Hub as a proper Dataset.

    Args:
        output_file: Path to the local JSONL file.
        repo_name: HuggingFace repo name (e.g., 'Codemaster67/cleanmol-pretrain-tasks').
        token: HuggingFace API token.
        test_size: Fraction of data to use as test split.
    """
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    print(f"\nPreparing dataset for HuggingFace Hub upload...")

    # Load JSONL into a list of dicts
    records = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  Loaded {len(records)} records from {output_file}")

    # Create HF Dataset
    ds = Dataset.from_list(records)

    # Train/test split
    split = ds.train_test_split(test_size=test_size, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "test": split["test"],
    })

    print(f"  Train: {len(dataset_dict['train'])} examples")
    print(f"  Test:  {len(dataset_dict['test'])} examples")

    # Push to Hub
    print(f"  Pushing to https://huggingface.co/datasets/{repo_name} ...")
    dataset_dict.push_to_hub(
        repo_name,
        token=token,
        private=False,
    )
    print(f"  Successfully pushed to HuggingFace Hub!")
    print(f"  URL: https://huggingface.co/datasets/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate CLEANMOL data")
    parser.add_argument("--output_dir", type=str, default="./cleanmol_data")
    parser.add_argument("--num_molecules", type=int, default=50000,
                        help="Number of source molecules to process")
    parser.add_argument("--max_per_task", type=int, default=50000,
                        help="Max examples per task type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push generated dataset to HuggingFace Hub")
    parser.add_argument("--hf_repo", type=str,
                        default="Codemaster67/cleanmol-pretrain-tasks",
                        help="HuggingFace dataset repo name")
    parser.add_argument("--hf_token", type=str,
                        default="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",
                        help="HuggingFace API token")
    parser.add_argument("--test_size", type=float, default=0.05,
                        help="Fraction of data for test split (default: 5%%)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load molecules
    print(f"Loading {args.num_molecules} molecules...")
    smiles_list = load_molecules(args.num_molecules)

    # Generate tasks
    print(f"Generating CLEANMOL tasks from {len(smiles_list)} molecules...")
    all_tasks = generate_all_tasks(smiles_list, max_per_task=args.max_per_task)

    # Shuffle
    random.shuffle(all_tasks)

    # Save locally
    output_file = os.path.join(args.output_dir, "cleanmol_tasks.jsonl")
    with open(output_file, "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task) + "\n")

    print(f"\n=== Summary ===")
    print(f"Total tasks generated: {len(all_tasks)}")
    task_counts = defaultdict(int)
    for t in all_tasks:
        task_counts[t["task"]] += 1
    for k, v in sorted(task_counts.items()):
        print(f"  {k}: {v}")
    print(f"Saved to: {output_file}")

    # Push to HuggingFace Hub
    if args.push_to_hub:
        push_dataset_to_hub(
            output_file,
            repo_name=args.hf_repo,
            token=args.hf_token,
            test_size=args.test_size,
        )
    else:
        print("\nTip: Add --push_to_hub to upload to HuggingFace Hub.")


if __name__ == "__main__":
    main()
