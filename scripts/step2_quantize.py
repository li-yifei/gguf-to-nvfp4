#!/usr/bin/env python3
# step2_quantize.py -- NVFP4 quantization via llmcompressor oneshot
import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    parser = argparse.ArgumentParser(
        description="Quantize an HF model to NVFP4 using llmcompressor"
    )
    parser.add_argument("--model-dir", required=True,
                        help="Path to bf16 HF model directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for quantized model")
    parser.add_argument("--num-calibration-samples", type=int, default=512,
                        help="Number of calibration samples")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length for calibration")
    parser.add_argument("--dataset", default="neuralmagic/calibration",
                        help="HF dataset for calibration")
    parser.add_argument("--dataset-config", default="LLM",
                        help="Dataset configuration/subset name")
    args = parser.parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=False,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets=["Linear"],
        ignore=["lm_head", "re:.*visual.*", "re:.*in_proj_a$", "re:.*in_proj_b$"],
        scheme="NVFP4",
    )

    print("Loading calibration dataset...")
    ds = load_dataset(args.dataset, args.dataset_config, split="train")

    print("Running NVFP4 oneshot quantization with save_compressed=True...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=ds,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        output_dir=args.output_dir,
        save_compressed=True,
    )

    print(f"\nQuantization complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
