#!/usr/bin/env python3
"""step2_quantize_gemma4_e4b.py -- NVFP4A16 quantization for Gemma 4 E4B.

Weight-only NVFP4 (activations stay BF16). Uses NVFP4A16 instead of full
NVFP4 (w4a4) because E4B's architecture (per-layer embeddings, audio
encoder, dynamic masking) breaks fx.symbolic_trace, which is required
for the sequential calibration pipeline that w4a4 needs.

NVFP4A16 quantizes weights from their own min/max statistics -- no data
flow, no calibration dataset, no tracing. The 4-bit weight quantization
dominates the error floor anyway, so the quality impact is small.

Vision tower, audio tower, embed_vision, embed_audio, and lm_head are
kept in BF16 via the ignore list. llmcompressor handles the multimodal
model in a single oneshot call and writes a complete
Gemma4ForConditionalGeneration checkpoint -- no separate stitch step.
"""

import argparse

from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    p = argparse.ArgumentParser(
        description="NVFP4A16 quantize a Gemma 4 E4B multimodal model"
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Path to the BF16 HF model directory (output of step1)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--sanity-prompt",
        default="What is the capital of France? Answer in one word.",
        help="Prompt used for a quick post-quantization generation check",
    )
    p.add_argument(
        "--sanity-max-new-tokens",
        type=int,
        default=20,
    )
    args = p.parse_args()

    print(f"[load] {args.model_dir}")
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype="auto")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(
        f"[load] params={n_params:.1f}B dtype={next(model.parameters()).dtype}"
    )

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4A16",
        ignore=[
            "lm_head",
            "re:.*vision_tower.*",
            "re:.*audio_tower.*",
            "re:.*embed_vision.*",
            "re:.*embed_audio.*",
        ],
    )

    print("[oneshot] starting NVFP4A16 (weight-only, no calibration)")
    oneshot(model=model, recipe=recipe)
    print("[oneshot] done")

    print("\n========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    messages = [{"role": "user", "content": args.sanity_prompt}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=args.sanity_max_new_tokens)
    print(processor.decode(output[0], skip_special_tokens=True))
    print("==========================================\n")

    print(f"[save] -> {args.output_dir}")
    model.save_pretrained(args.output_dir, save_compressed=True)
    processor.save_pretrained(args.output_dir)
    print("[OK] Gemma 4 E4B NVFP4A16 quantization complete")


if __name__ == "__main__":
    main()
