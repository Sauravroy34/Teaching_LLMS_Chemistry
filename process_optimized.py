# ============================================================
# OFFLINE DATASET PREPARATION FOR OLMO-7B MULTI-OBJECTIVE
# RUN THIS SCRIPT ONCE BEFORE TRAINING
# ============================================================
# OPTIMIZED: Fixes RDKit memory leak (github.com/rdkit/rdkit/issues/3239)
# Strategy: 
#   1) RDKit processing uses multiprocessing.Pool with maxtasksperchild
#      so workers restart periodically, reclaiming all C++ heap memory.
#   2) Work is split into shards with explicit GC between them.
#   3) Tokenization (no RDKit) uses HF datasets.map() with num_proc.
#   4) Per-molecule timeout prevents BRICSDecompose from hanging.
# ============================================================
import gc
import os
import random
import signal
import sys
import time
import multiprocessing
from functools import partial

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from huggingface_hub import login

# ============================================================
# CRITICAL: Prevent tokenizer's internal Rust thread pool from
# conflicting with our multiprocessing (causes deadlock + bloat)
# ============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "allenai/OLMo-7B-hf"
MAX_LEN = 256
OUTPUT_DIR = "h100_ready_pubchem"
NUM_SAMPLES = 5_000_000

# --- Memory-aware tuning ---
NUM_CORES = 30
SHARD_SIZE = 250_000        # Molecules per shard (controls peak RAM)
WORKER_LIFETIME = 5_000     # Restart worker after this many molecules (kills C++ heap)
WRITER_BATCH_SIZE = 500     # Flush Arrow buffers to disk more frequently

# --- Safety limits ---
MAX_HEAVY_ATOMS = 100       # Skip molecules larger than this (prevents BRICS explosion)
PER_MOL_TIMEOUT = 30        # Seconds before we abandon a single molecule

# --- Hugging Face Hub ---
HF_TOKEN = "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"
HF_REPO_NAME = "pubchem-5m-multi-objective"  # Will be pushed as <your_username>/pubchem-5m-multi-objective

# ============================================================
# TOKENIZER INITIALIZATION (only used in main process for step 2)
# ============================================================

print(f"Loading tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# TIMEOUT HELPER — used inside worker subprocesses (Unix only)
# ============================================================

class MoleculeTimeout(Exception):
    """Raised when a single molecule exceeds PER_MOL_TIMEOUT."""
    pass

def _timeout_handler(signum, frame):
    raise MoleculeTimeout("Molecule processing timed out")


# ============================================================
# WORKER FUNCTION — runs in subprocess
# ============================================================
# Each call to this function processes one SMILES string.
# Workers are recycled after WORKER_LIFETIME calls via
# maxtasksperchild, which kills the subprocess and frees
# ALL C++ heap memory accumulated by RDKit.
# ============================================================

def _process_one(smiles):
    """Process a single SMILES → text string.
    
    RDKit imports happen once per worker process (cached by Python).
    The mol object is created and destroyed within this call.
    When the worker subprocess is killed (after maxtasksperchild),
    ALL accumulated C++ memory is reclaimed by the OS.
    
    Safety features:
      - signal.alarm(PER_MOL_TIMEOUT) kills hung BRICSDecompose calls
      - Molecules with >MAX_HEAVY_ATOMS are skipped (prevents combinatorial explosion)
    """
    import selfies as sf
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.BRICS import BRICSDecompose
    
    # Set up per-molecule timeout (Unix signal-based)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(PER_MOL_TIMEOUT)
    
    try:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            return None
        
        if mol is None:
            return None
        
        # --- Skip overly complex molecules (BRICS will explode) ---
        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy > MAX_HEAVY_ATOMS:
            del mol
            return None

        mode = random.choice([
            "causal", "scaffold_mask", "brics_mask", "view_prediction"
        ])

        try:
            if mode == "causal":
                rand_views = []
                for _ in range(3):
                    try:
                        rand_views.append(
                            Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                        )
                    except Exception:
                        rand_views.append("")
                
                try:
                    selfies_str = sf.encoder(smiles)
                except Exception:
                    selfies_str = ""
                
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                except Exception:
                    scaffold = ""
                
                try:
                    brics = list(BRICSDecompose(mol))
                except Exception:
                    brics = []
                
                tags = []
                try:
                    if any(a.GetIsAromatic() for a in mol.GetAtoms()):
                        tags.append("aromatic")
                    if mol.GetRingInfo().NumRings() > 0:
                        tags.append("ring")
                except Exception:
                    pass
                
                text = (
                    f"\n<SMILES>\n{smiles}\n\n"
                    f"<RANDOM_SMILES>\n{' '.join(rand_views)}\n\n"
                    f"<SELFIES>\n{selfies_str}\n\n"
                    f"<BRICS>\n{' '.join(brics[:10])}\n\n"
                    f"<SCAFFOLD>\n{scaffold}\n\n"
                    f"<TAGS>\n{' '.join(tags)}\n"
                )
            
            elif mode == "scaffold_mask":
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                except Exception:
                    scaffold = ""
                
                text = (
                    f"\n<SMILES>\n{smiles}\n\n"
                    f"<SCAFFOLD>\n[MASK]\n\n"
                    f"<ANSWER>\n{scaffold}\n"
                )
            
            elif mode == "brics_mask":
                try:
                    brics = list(BRICSDecompose(mol))
                except Exception:
                    brics = []
                
                text = (
                    f"\n<SMILES>\n{smiles}\n\n"
                    f"<BRICS>\n[MASK]\n\n"
                    f"<ANSWER>\n{' '.join(brics[:10])}\n"
                )
            
            else:  # view_prediction
                try:
                    rand_view = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                except Exception:
                    rand_view = ""
                
                text = (
                    f"\n<CANONICAL>\n{smiles}\n\n"
                    f"<RANDOMIZED>\n{rand_view}\n"
                )
        finally:
            del mol
        
        return text
    
    except MoleculeTimeout:
        # This molecule took too long (likely BRICSDecompose on a complex mol)
        return None
    except Exception:
        return None
    finally:
        # Cancel any pending alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================
# TOKENIZATION (HuggingFace datasets — no RDKit, safe for num_proc)
# ============================================================

def tokenize_batched(examples):
    out = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    out["labels"] = [ids.copy() for ids in out["input_ids"]]
    return out


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # Log in to Hugging Face Hub
    login(token=HF_TOKEN)
    
    print(f"Starting data processing using {NUM_CORES} CPU cores...")
    print(f"Shard size: {SHARD_SIZE:,} | Worker lifetime: {WORKER_LIFETIME:,} molecules")
    print(f"Per-molecule timeout: {PER_MOL_TIMEOUT}s | Max heavy atoms: {MAX_HEAVY_ATOMS}")

    print("Downloading/Loading raw dataset...")
    raw_dataset = load_dataset(
        "sagawa/pubchem-10m-canonicalized", 
        split="train"
    )

    print(f"Selecting {NUM_SAMPLES:,} samples...")
    raw_dataset = raw_dataset.select(range(NUM_SAMPLES))
    
    # Extract all SMILES to a plain list (lightweight — just strings)
    all_smiles = raw_dataset["smiles"]
    
    # Free the HF dataset object — we'll rebuild from processed text
    del raw_dataset
    gc.collect()

    # ================================================================
    # STEP 1: RDKIT PROCESSING WITH SUBPROCESS ISOLATION
    # ================================================================
    # multiprocessing.Pool with maxtasksperchild=WORKER_LIFETIME
    # Each worker process is KILLED and RESPAWNED after processing
    # WORKER_LIFETIME molecules. This is the key fix:
    #   - RDKit's C++ allocator fragments memory over time
    #   - del mol + gc.collect() don't reclaim the C++ heap
    #   - Killing the subprocess reclaims ALL memory (OS-level)
    #
    # Additional safety:
    #   - signal.alarm(30) prevents any single mol from blocking
    #   - Molecules with >100 heavy atoms are skipped entirely
    # ================================================================
    
    num_shards = (NUM_SAMPLES + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"\nStep 1/2: Processing {NUM_SAMPLES:,} molecules in {num_shards} shards...")
    
    all_texts = []
    total_valid = 0
    total_invalid = 0
    total_timed_out = 0
    t_start = time.time()
    
    for shard_idx in range(num_shards):
        start = shard_idx * SHARD_SIZE
        end = min(start + SHARD_SIZE, NUM_SAMPLES)
        shard_smiles = all_smiles[start:end]
        
        shard_t = time.time()
        shard_valid = 0
        shard_done = 0
        
        # Create a FRESH pool for each shard with maxtasksperchild
        # Workers auto-restart after WORKER_LIFETIME tasks
        with multiprocessing.Pool(
            processes=NUM_CORES, 
            maxtasksperchild=WORKER_LIFETIME
        ) as pool:
            # imap_unordered for best throughput — order doesn't matter
            # chunksize balances IPC overhead vs. memory per worker
            for text in pool.imap_unordered(
                _process_one, 
                shard_smiles, 
                chunksize=200  
            ):
                shard_done += 1
                if text is not None:
                    all_texts.append(text)
                    shard_valid += 1
                
                # --- Live progress every 10,000 molecules ---
                if shard_done % 10_000 == 0:
                    elapsed = time.time() - shard_t
                    rate = shard_done / elapsed if elapsed > 0 else 0
                    remaining = len(shard_smiles) - shard_done
                    shard_eta = remaining / rate if rate > 0 else 0
                    print(f"    Shard {shard_idx+1}: {shard_done:,}/{len(shard_smiles):,} "
                          f"({shard_valid:,} valid) "
                          f"[{rate:.0f} mol/s, ETA {shard_eta:.0f}s]",
                          flush=True)
        
        invalid_count = len(shard_smiles) - shard_valid
        total_valid += shard_valid
        total_invalid += invalid_count
        
        # Free shard data
        del shard_smiles
        gc.collect()
        
        elapsed = time.time() - shard_t
        total_elapsed = time.time() - t_start
        rate = (end) / total_elapsed
        eta = (NUM_SAMPLES - end) / rate if rate > 0 else 0
        
        print(f"  ✅ Shard {shard_idx + 1}/{num_shards} "
              f"[{start:,}-{end:,}] "
              f"✓{shard_valid:,} ✗{invalid_count:,} "
              f"({elapsed:.1f}s) "
              f"ETA: {eta/60:.1f}min",
              flush=True)
    
    del all_smiles
    gc.collect()
    
    total_time_step1 = time.time() - t_start
    print(f"\nRDKit processing complete: {total_valid:,} valid, "
          f"{total_invalid:,} invalid ({total_time_step1/60:.1f} min)")

    # ================================================================
    # STEP 2: BUILD HF DATASET + TOKENIZE
    # ================================================================
    # No RDKit involved — safe to use HF datasets with num_proc.
    # ================================================================
    
    print(f"\nStep 2/2: Building dataset and tokenizing...")
    
    # Create HF dataset from the text list
    processed_dataset = Dataset.from_dict({"text": all_texts})
    
    # Free the list
    del all_texts
    gc.collect()
    
    # --- Push non-tokenized dataset to Hugging Face Hub ---
    print(f"Pushing non-tokenized dataset to HF Hub: {HF_REPO_NAME}...")
    processed_dataset.push_to_hub(
        HF_REPO_NAME,
        token=HF_TOKEN,
        private=False,
    )
    print("✅ Non-tokenized dataset pushed to Hugging Face Hub!")
    
    # Tokenize with full parallelism (no RDKit, no memory leak risk)
    processed_dataset = processed_dataset.map(
        tokenize_batched, 
        batched=True,
        batch_size=5000,
        num_proc=NUM_CORES,
        remove_columns=["text"],
        writer_batch_size=WRITER_BATCH_SIZE,
    )

    print(f"Saving fully processed dataset to ./{OUTPUT_DIR}...")
    processed_dataset.save_to_disk(OUTPUT_DIR)

    print(f"\n✅ Dataset processing complete!")
    print(f"   Valid molecules: {total_valid:,}")
    print(f"   Total time: {(time.time() - t_start)/60:.1f} min")
    print(f"   Output: ./{OUTPUT_DIR}/")
