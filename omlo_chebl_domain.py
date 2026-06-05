"""
2×2 LS (Layer-Selective) Continual Pre-training on ChEBI-20-MM
================================================================

Strategy from Yamaguchi et al. (2026) "How Can We Effectively Expand the
Vocabulary of LLMs with 0.01GB of Target Language Text?"

2×2 LS = Train only the top and bottom 2 transformer layers + embeddings + LM head.
This calibrates the encoding/decoding layers closest to the input/output while
minimizing catastrophic forgetting in the model's core knowledge layers.

Key features:
  - 2×2 LS selective layer unfreezing
  - Causal Language Modeling (CLM) objective
  - Template randomization (5 templates) to prevent positional shortcut learning
  - Packing (concatenate + chunk) for efficient training
  - SMILES wrapped in <|start_of_smiles|> ... <|end_of_smiles|>
  - Validation with perplexity tracking
  - 3 epochs, lr=1e-4, seq_len=512
  - Push to HuggingFace Hub on completion
"""

import os
import math
import random
import torch
import logging
from functools import partial
from itertools import chain

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
MODEL_NAME = "allenai/OLMo-7B-hf"          # Source model
DATASET_NAME = "liupf/ChEBI-20-MM"          # ChEBI-20 multi-modal
OUTPUT_DIR = "./olmo_2x2ls_chebi20"
HF_REPO_ID = "Codemaster67/OLmo-chebl_domain_adaption"
SEED = 42

# Training hyperparameters
# - seq_len=512: paper §7.2 shows shorter seq → more gradient updates → avoids underfitting
# - batch=4, accum=2: effective batch=8, matching paper §6.6
# - lr=1e-4: matches paper §6.6
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 4                              # Per-device (A100-40GB fits this with seq_len=512)
GRADIENT_ACCUMULATION_STEPS = 2             # Effective batch = 4 * 2 = 8
MAX_SEQ_LENGTH = 512                        # Paper recommends 512 for low-resource
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
EVAL_STEPS = 50                             # Adjusted for smaller total steps
SAVE_STEPS = 50

# Special tokens for SMILES encapsulation
SMILES_START = "<|start_of_smiles|>"
SMILES_END = "<|end_of_smiles|>"

# ─────────────────────────────────────────────────────────
# Template Pool for Randomization
# ─────────────────────────────────────────────────────────
# Varying the position of SMILES relative to text prevents the model
# from learning positional shortcuts (e.g., "SMILES is always at position 0").
# This forces the model to learn genuine SMILES↔text semantic associations
# rather than memorizing a fixed template structure.
# Inspired by MolXPT (Liu et al. 2023) which wraps SMILES inline in varying
# text contexts and achieves superior molecular representations.
TEMPLATES = [
    # 1. SMILES first (classic pre-training / raw association)
    "{som}{smiles}{eom}\n{description}",

    # 2. Middle wrapping (MolXPT-inspired inline context)
    "The chemical structure {som}{smiles}{eom} can be described as follows: {description}",

    # 3. Text first, SMILES last (generation style)
    "{description} This compound is represented by the structure {som}{smiles}{eom}.",

    # 4. SMILES first with connector (property prediction style)
    "Regarding {som}{smiles}{eom}, {description}",

    # 5. Conversational / integrative phrasing
    "Based on its properties, {description} Structurally, we map this to {som}{smiles}{eom}.",
]


# ─────────────────────────────────────────────────────────
# 1. Tokenizer Setup
# ─────────────────────────────────────────────────────────
def setup_tokenizer(tokenizer_id = "Codemaster67/OLMO_Smiles_aware_tokenizer") -> AutoTokenizer:
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )

    return tokenizer



# ─────────────────────────────────────────────────────────
# 2. Model Setup with 2×2 LS Freezing
# ─────────────────────────────────────────────────────────
def setup_model(model_id = "allenai/OLMo-7B-hf",new_tokenizer_id = "Codemaster67/OLMO_Smiles_aware_tokenizer",token_file = "spe_codes.txt" ) -> AutoModelForCausalLM:
    """
    Load model and apply 2×2 LS (Layer-Selective) freezing strategy.

    Trainable parameters:
      - Bottom 2 transformer layers (layers 0, 1)
      - Top 2 transformer layers (layers N-2, N-1)
      - Embedding layer (input)
      - Language modeling head (output / lm_head)

    Frozen parameters:
      - All middle transformer layers (layers 2 ... N-3)
    """

    print("Loading tokenizers and model...")
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True,
                                                attn_implementation="flash_attention_2",
                                                )


    new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_id)
    # 2. LOAD YOUR NEW SMILES PAIR TOKENS
    # Assuming your text file has one token per line
    token_file = "spe_codes.txt"
    with open(token_file, "r") as f:
        new_smiles_tokens = [line.strip() for line in f if line.strip()]

    # Resize the model's embedding matrix to fit the new tokens

    new_smiles_tokens = new_smiles_tokens + [SMILES_START,SMILES_END]

    print(len(new_smiles_tokens))
    model.resize_token_embeddings(len(new_tokenizer))

    # 4. PERFORM MEAN INITIALIZATION
    # Get the pointer to the model's actual embedding weights
    embeddings = model.get_input_embeddings().weight.data

    print("Starting Mean Initialization...")
    with torch.no_grad():
        for token in new_smiles_tokens:
            
            # STEP A: Break the new token down using the UNMODIFIED base tokenizer
            # E.g., "C=C" -> IDs for ["C", "=", "C"]
            sub_token_ids = base_tokenizer.encode(token, add_special_tokens=False)
            
            if len(sub_token_ids) > 0:
                # STEP B: Fetch the embedding vectors for those constituent sub-tokens
                sub_token_vectors = embeddings[sub_token_ids]
                
                # STEP C: Calculate the mean (average) across the rows (dim=0)
                mean_vector = torch.mean(sub_token_vectors, dim=0)
                
                # STEP D: Find where the new token lives in the EXPANDED tokenizer
                new_target_id = tokenizer.convert_tokens_to_ids(token)
                
                # STEP E: Overwrite the random noise with our calculated mean vector
                embeddings[new_target_id] = mean_vector

    print("Mean initialization complete! Model is ready for finetuning.")
    # ── Apply 2×2 LS freezing ──
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Identify transformer layers
    # OLMo uses model.model.layers (list of transformer blocks)
    if hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model.model, "transformer"):
        layers = model.model.transformer.h
    else:
        raise ValueError(
            "Cannot identify transformer layers. "
            "Please check model architecture."
        )

    num_layers = len(layers)
    logger.info(f"Model has {num_layers} transformer layers")

    # Bottom 2 layers (closest to embedding / input encoding)
    bottom_layers = [0, 1]
    # Top 2 layers (closest to LM head / output decoding)
    top_layers = [num_layers - 2, num_layers - 1]
    trainable_layer_indices = set(bottom_layers + top_layers)

    logger.info(
        f"2×2 LS: Unfreezing layers {sorted(trainable_layer_indices)} "
        f"(bottom={bottom_layers}, top={top_layers})"
    )

    # Step 3: Unfreeze selected layers
    for idx in trainable_layer_indices:
        for param in layers[idx].parameters():
            param.requires_grad = True

    # Step 4: Unfreeze embeddings (input)
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True

    # Step 5: Unfreeze LM head (output)
    if model.get_output_embeddings() is not None:
        for param in model.get_output_embeddings().parameters():
            param.requires_grad = True

    # Step 6: Unfreeze final layer norm (important for output calibration)
    if hasattr(model.model, "norm"):
        for param in model.model.norm.parameters():
            param.requires_grad = True
    elif hasattr(model.model, "ln_f"):
        for param in model.model.ln_f.parameters():
            param.requires_grad = True

    # ── Report parameter counts ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    return model


# ─────────────────────────────────────────────────────────
# 3. Dataset Processing with Template Randomization
# ─────────────────────────────────────────────────────────
def format_sample(example: dict) -> dict:
    """
    Format a ChEBI-20-MM sample into text for CLM training.

    Randomly selects one of 5 templates to place SMILES in different
    positions relative to the description text. This prevents the model
    from learning positional shortcuts and forces genuine semantic
    association between SMILES and text.

    Templates vary SMILES position: beginning, middle, end, and
    integrated within conversational phrasing.
    """
    smiles = example.get("SMILES", "")
    description = example.get("description", "")

    # Randomly select a template
    template = random.choice(TEMPLATES)

    # Fill in the template
    text = template.format(
        som=SMILES_START,
        eom=SMILES_END,
        smiles=smiles,
        description=description,
    )

    return {"text": text}


def tokenize_function(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenize text ."""
    # Add EOS to each example so the model learns sequence boundaries
    texts = [t for t in examples["text"]]
    return tokenizer(
        texts,
        truncation=False,  # Don't truncate — packing handles length
        return_attention_mask=False,
    )


def pack_sequences(tokenized_dataset, tokenizer, max_seq_length: int):
    """
    Pack (concatenate + chunk) tokenized sequences into fixed-length blocks.

    This is the standard "packing" approach for CLM:
      1. Concatenate all token sequences with EOS separators
      2. Chunk into blocks of max_seq_length
      3. Labels = shifted input_ids (handled by DataCollatorForLanguageModeling)

    Benefits:
      - No padding waste → maximum GPU utilization
      - Each batch element contains ~max_seq_length real tokens
      - Multiple short examples are packed into one sequence
    """

    def group_texts(examples):
        # Concatenate all tokenized texts
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # Drop the remainder that doesn't fill a complete block
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length

        # Split into chunks of max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }

        # For CLM, labels are the same as input_ids (shifted internally by the model)
        result["labels"] = result["input_ids"].copy()
        return result

    packed = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=tokenized_dataset.column_names,
        desc="Packing sequences",
    )

    return packed


def prepare_datasets(tokenizer: AutoTokenizer):
    """Load ChEBI-20-MM, format with randomized templates, tokenize, and pack."""

    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    # Dataset has train/validation/test splits
    logger.info(f"Dataset splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        logger.info(f"  {split_name}: {len(split_data)} samples")

    # Format: add SMILES encapsulation + description with randomized templates
    logger.info("Formatting samples with randomized SMILES templates...")

    # Seed RNG for reproducibility before formatting
    random.seed(SEED)

    formatted = dataset.map(
        format_sample,
        remove_columns=[c for c in dataset["train"].column_names if c != "text"],
        desc="Formatting with template randomization",
    )

    # Log template distribution from a sample
    logger.info("Sample formatted texts (one per template style):")
    for i in range(min(5, len(formatted["train"]))):
        logger.info(f"  [{i}] {formatted['train'][i]['text'][:120]}...")

    # Tokenize
    logger.info("Tokenizing...")
    tokenize_fn = partial(tokenize_function, tokenizer=tokenizer)
    tokenized = formatted.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Pack sequences for efficient training
    logger.info(f"Packing sequences to length {MAX_SEQ_LENGTH}...")
    train_packed = pack_sequences(tokenized["train"], tokenizer, MAX_SEQ_LENGTH)
    val_packed = pack_sequences(tokenized["validation"], tokenizer, MAX_SEQ_LENGTH)

    logger.info(f"Packed training sequences:   {len(train_packed)}")
    logger.info(f"Packed validation sequences: {len(val_packed)}")

    # Log a sample to verify formatting
    sample_ids = train_packed[0]["input_ids"][:200]
    logger.info(f"Sample packed text (first 200 tokens):\n{tokenizer.decode(sample_ids)}")

    return train_packed, val_packed


# ─────────────────────────────────────────────────────────
# 4. Custom Trainer with Perplexity Logging
# ─────────────────────────────────────────────────────────
class CLMTrainerWithPerplexity(Trainer):
    """Extends HuggingFace Trainer to log perplexity during evaluation."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Compute perplexity from eval loss
        eval_loss_key = f"{metric_key_prefix}_loss"
        if eval_loss_key in metrics:
            try:
                perplexity = math.exp(metrics[eval_loss_key])
            except OverflowError:
                perplexity = float("inf")
            metrics[f"{metric_key_prefix}_perplexity"] = perplexity
            logger.info(f"Perplexity: {perplexity:.2f}")

        return metrics


# ─────────────────────────────────────────────────────────
# 5. Main Training Loop
# ─────────────────────────────────────────────────────────
def main():
    set_seed(SEED)

    # Setup tokenizer
    tokenizer = setup_tokenizer()

    # Setup model with 2×2 LS freezing
    model = setup_model()

    # Prepare datasets (format with template randomization → tokenize → pack)
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    # Data collator for CLM (no masking, just handles labels alignment)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Training schedule
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimizer
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        # Precision
        bf16=True,
        tf32=True,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        save_steps=SAVE_STEPS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",  # Set to "wandb" if you want W&B tracking

        # Memory optimization
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # HuggingFace Hub push
        push_to_hub=True,
        hub_model_id=HF_REPO_ID,
        hub_strategy="end",  # Push only at the end of training

        # Misc
        seed=SEED,
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = CLMTrainerWithPerplexity(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ── Run initial evaluation (baseline) ──
    logger.info("Running baseline evaluation before training...")
    baseline_metrics = trainer.evaluate()
    logger.info(f"Baseline metrics: {baseline_metrics}")

    # ── Train ──
    logger.info("Starting 2×2 LS training with template randomization...")
    train_result = trainer.train()

    # ── Save final model locally ──
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Push to HuggingFace Hub ──
    logger.info(f"Pushing model to HuggingFace Hub: {HF_REPO_ID}")
    try:
        trainer.push_to_hub(
            commit_message=(
                "2×2 LS domain adaptation on ChEBI-20-MM\n\n"
                "Strategy: 2×2 LS (top 2 + bottom 2 layers)\n"
                "Objective: Causal Language Modeling (CLM)\n"
                "Template: 5-template randomization for SMILES position\n"
                f"Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}, SeqLen: {MAX_SEQ_LENGTH}\n"
                f"Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n"
                f"Special tokens: {SMILES_START}, {SMILES_END}"
            ),
        )
        logger.info(f"Successfully pushed to {HF_REPO_ID}")
    except Exception as e:
        logger.error(f"Failed to push to Hub: {e}")
        logger.info("Model is saved locally. You can push manually with:")
        logger.info(f"  trainer.push_to_hub() or huggingface-cli upload {HF_REPO_ID} {OUTPUT_DIR}")

    # ── Final evaluation ──
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  2×2 LS TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model:               {MODEL_NAME}")
    print(f"  Dataset:             {DATASET_NAME}")
    print(f"  Strategy:            2×2 LS (top 2 + bottom 2 layers)")
    print(f"  Objective:           Causal Language Modeling (CLM)")
    print(f"  Template pool:       {len(TEMPLATES)} randomized templates")
    print(f"  Packing:             Enabled (seq_len={MAX_SEQ_LENGTH})")
    print(f"  SMILES tokens:       {SMILES_START} / {SMILES_END}")
    print(f"  Epochs:              {NUM_EPOCHS}")
    print(f"  Learning rate:       {LEARNING_RATE}")
    print(f"  Effective batch:     {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  HuggingFace repo:    {HF_REPO_ID}")
    print("-" * 70)

    baseline_ppl = baseline_metrics.get("eval_perplexity", "N/A")
    final_ppl = final_metrics.get("eval_perplexity", "N/A")
    baseline_loss = baseline_metrics.get("eval_loss", "N/A")
    final_loss = final_metrics.get("eval_loss", "N/A")

    print(f"  Baseline eval loss:  {baseline_loss}")
    print(f"  Baseline perplexity: {baseline_ppl}")
    print(f"  Final eval loss:     {final_loss}")
    print(f"  Final perplexity:    {final_ppl}")
    print(f"  Train loss:          {train_result.training_loss:.4f}")
    print(f"  Output saved to:     {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

