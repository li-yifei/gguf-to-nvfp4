#!/usr/bin/env python3
"""step2_quantize_qwen36_moe.py -- NVFP4 quantization for Qwen3.6-35B-A3B MoE.

Uses llmcompressor oneshot with MoE-aware calibration.
Conservative recipe following AEON-7/RedHatAI approach: linear_attn (30
DeltaNet/Mamba layers) kept in bf16 for quality, MTP kept in bf16 for
speculative decoding. Only full-attention projections and MoE experts
are quantized to NVFP4.

Key MoE adaptations vs Qwen3.5 dense:
- Qwen3_5MoeForConditionalGeneration (not AutoModelForCausalLM)
- moe_calibrate_all_experts=True (all 256 experts see calibration data)
- Ignore router gates, shared expert gate, linear_attn, MTP, visual
"""

import argparse

from datasets import load_dataset
from transformers import Qwen3_5MoeForConditionalGeneration, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier




def main():
    p = argparse.ArgumentParser(
        description="Quantize Qwen3.6-35B-A3B HF model to NVFP4"
    )
    p.add_argument("--model-dir", required=True,
                   help="Path to bf16 HF model directory (from step1)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for NVFP4 model")
    p.add_argument("--num-samples", type=int, default=256,
                   help="Number of calibration samples (AEON-7 uses 256)")
    p.add_argument("--max-seq-length", type=int, default=2048,
                   help="Maximum sequence length for calibration")
    args = p.parse_args()

    # Use device_map="auto" with disk offload for MoE expert format conversion.
    # save_pretrained bug (AttributeError on get_submodule) is fixed by patching
    # transformers/integrations/accelerate.py to skip non-matching submodule paths.
    print("Loading model with Qwen3_5MoeForConditionalGeneration...")
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

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "lm_head",
            "re:.*visual.*",
            "re:.*mlp.gate$",
            "re:.*mlp.shared_expert_gate$",
            "re:.*linear_attn.*",
            "re:^mtp.*",
        ],
    )

    print("Loading calibration dataset (ultrachat_200k)...")
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

    print(f"Running NVFP4 oneshot ({args.num_samples} samples, "
          f"seq_len={args.max_seq_length}, moe_calibrate_all_experts=True)...")
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
