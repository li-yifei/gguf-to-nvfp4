#!/usr/bin/env python3
"""step2b_quantize_aggressive.py -- Aggressive NVFP4 for 100K context on 32GB.

Sakamakismile approach: quantize EVERYTHING except lm_head + gates + visual.
This includes linear_attn (DeltaNet) and MTP layers → smaller model footprint
→ more VRAM for KV cache → longer context.

Target: model ~22GB → 10GB free for fp8 KV cache → ~100K context on RTX 5090.
"""

import argparse

from datasets import load_dataset
from transformers import Qwen3_5MoeForConditionalGeneration, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    p = argparse.ArgumentParser(
        description="Aggressive NVFP4 quantization for max context length"
    )
    p.add_argument("--model-dir", required=True,
                   help="Path to bf16 HF model directory (from step1)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for NVFP4 model")
    p.add_argument("--num-samples", type=int, default=128,
                   help="Number of calibration samples")
    p.add_argument("--max-seq-length", type=int, default=1024,
                   help="Maximum sequence length for calibration")
    args = p.parse_args()

    print("Loading model...")
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model_dir,
        dtype="auto",
        device_map="auto",
        max_memory={0: "4GiB"},
        offload_folder="/tmp/qwen36-offload",
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True)

    # Sakamakismile recipe: only exclude lm_head + vision + gates
    # Everything else (linear_attn, MTP) gets NVFP4 → smaller footprint
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "lm_head",
            "re:.*visual.*",
            "re:.*mlp.gate$",
            "re:.*mlp.shared_expert_gate$",
        ],
    )

    print("Loading calibration dataset...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                      split=f"train_sft[:{args.num_samples}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"], padding=False,
            max_length=args.max_seq_length, truncation=True,
            add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    print(f"Running aggressive NVFP4 ({args.num_samples} samples, "
          f"seq_len={args.max_seq_length})...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
        output_dir=args.output_dir,
        save_compressed=True,
        moe_calibrate_all_experts=True,
        pipeline="basic",
    )

    print(f"\nQuantization complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
