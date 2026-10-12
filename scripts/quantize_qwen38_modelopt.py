#!/usr/bin/env python3
"""Quantize a local Qwen3.8 BF16 VLM using the historical ModelOpt recipe."""
import argparse
import os
from pathlib import Path
import shutil

from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.export.model_utils import get_language_model_from_vl
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--calib-size", type=int, default=20)
    p.add_argument("--calib-seq", type=int, default=8192)
    p.add_argument("--gpu-budget", default="4GiB")
    p.add_argument("--cpu-budget", default="52GiB")
    p.add_argument("--offload", required=True)
    p.add_argument("--dataset", default="cnn_dailymail")
    args = p.parse_args()

    if not os.path.isdir(args.model_id):
        p.error("--model-id must be a local, revision-pinned BF16 checkpoint")
    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        p.error("output directory must be empty")
    if args.calib_size < 1 or args.calib_seq < 1:
        p.error("calibration sample count and sequence length must be positive")
    os.makedirs(args.offload, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/5] Load full VL model: {args.model_id}")
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map="auto",
        max_memory={0: args.gpu_budget, "cpu": args.cpu_budget},
        offload_folder=args.offload,
        trust_remote_code=True,
    )

    print("[2/5] Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print("[3/5] Extract language_model from VL")
    lineage = get_language_model_from_vl(full_model)
    language_model = lineage[-1] if lineage else full_model
    print(f"    found language_model via {len(lineage) if lineage else 0} levels")

    # Preserve the historical language-only quantization exclusions.
    quant_cfg = {**mtq.NVFP4_DEFAULT_CFG}
    extra = [
        {"quantizer_name": "*lm_head*", "enable": False},
        {"quantizer_name": "*linear_attn.conv1d*", "enable": False},
        {"quantizer_name": "*mtp.*", "enable": False},
        # 多模态保留 vision tower
        {"quantizer_name": "*visual.*", "enable": False},
        {"quantizer_name": "*vision_model.*", "enable": False},
        {"quantizer_name": "*embed_vision.*", "enable": False},
    ]
    quant_cfg["quant_cfg"] = list(quant_cfg.get("quant_cfg", [])) + extra

    print(f"[4/5] Calibrate NVFP4 (samples={args.calib_size}, seq={args.calib_seq})")
    dl = get_dataset_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=1,
        num_samples=args.calib_size,
        max_sample_length=args.calib_seq,
        device=full_model.device,
    )
    calib_loop = create_forward_loop(dataloader=dl)
    mtq.quantize(language_model, quant_cfg, forward_loop=calib_loop)

    print(f"[5/5] Export HF checkpoint → {args.output_dir}")
    export_hf_checkpoint(full_model, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Retain multimodal processor assets from the exact quantized source.
    for name in ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        source = Path(args.model_id) / name
        destination = Path(args.output_dir) / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    print(f"\n✓ done → {args.output_dir}")


if __name__ == "__main__":
    main()
