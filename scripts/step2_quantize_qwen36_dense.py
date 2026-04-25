#!/usr/bin/env python3
"""NVFP4 quantization for HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive.

Follows the proven Qwen3.5-27B workflow in this repo:
- load the model manually with Transformers
- keep lm_head, visual tower, and linear-attn in_proj_a/b in bf16
- avoid oneshot unified saving to reduce OOM risk on 32 GB GPUs
- save sharded safetensors manually after calibration
"""

import argparse
import gc
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.abspath("./.hf_cache"))

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import get_token
import transformers.modeling_utils as modeling_utils
from transformers import AutoTokenizer, PreTrainedModel, Qwen3_5ForConditionalGeneration


if not hasattr(modeling_utils, "TORCH_INIT_FUNCTIONS"):
    fallback_init_names = [
        "uniform_",
        "normal_",
        "trunc_normal_",
        "constant_",
        "xavier_uniform_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "orthogonal_",
        "sparse_",
        "eye_",
        "dirac_",
    ]
    modeling_utils.TORCH_INIT_FUNCTIONS = {
        name: getattr(torch.nn.init, name)
        for name in fallback_init_names
        if hasattr(torch.nn.init, name)
    }

if not hasattr(PreTrainedModel, "_get_no_split_modules"):
    def _get_no_split_modules(self, device_map="auto"):
        del device_map
        return list(getattr(self, "_no_split_modules", []) or [])

    PreTrainedModel._get_no_split_modules = _get_no_split_modules

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


DEFAULT_IGNORE = [
    "lm_head",
    "re:.*visual.*",
    "re:.*linear_attn.in_proj_a$",
    "re:.*linear_attn.in_proj_b$",
]

COPY_AFTER_SAVE = [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
]


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive to NVFP4"
    )
    parser.add_argument(
        "--model-dir",
        default="./Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-HF",
        help="Local HF checkpoint directory for Qwen3.6-27B",
    )
    parser.add_argument(
        "--output-dir",
        default="./Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-NVFP4",
        help="Output directory for the NVFP4 checkpoint",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=64,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Calibration sequence length",
    )
    parser.add_argument(
        "--dataset",
        default="neuralmagic/calibration",
        help="Calibration dataset repo ID",
    )
    parser.add_argument(
        "--dataset-config",
        default="LLM",
        help="Calibration dataset config/subset",
    )
    parser.add_argument(
        "--dataset-mode",
        default="ultrachat_nemotron",
        choices=["ultrachat_nemotron", "ultrachat_openplatypus", "neuralmagic_llm"],
        help="Calibration dataset recipe to use",
    )
    parser.add_argument(
        "--ultrachat-samples",
        type=positive_int,
        default=None,
        help="Optional fixed count for UltraChat samples in mixed dataset modes",
    )
    parser.add_argument(
        "--secondary-samples",
        type=positive_int,
        default=None,
        help="Optional fixed count for the second dataset in mixed dataset modes",
    )
    parser.add_argument(
        "--gpu-memory",
        default="22GiB",
        help="Per-GPU memory budget passed to transformers",
    )
    parser.add_argument(
        "--cpu-memory",
        default="120GiB",
        help="CPU offload memory budget passed to transformers",
    )
    parser.add_argument(
        "--offload-dir",
        default=None,
        help="Optional disk offload directory",
    )
    return parser.parse_args()


def build_load_kwargs(args):
    load_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "ignore_mismatched_sizes": False,
    }

    offload_dir = args.offload_dir or os.path.join(args.output_dir, "_offload")
    os.makedirs(offload_dir, exist_ok=True)
    load_kwargs["offload_folder"] = offload_dir

    if torch.cuda.is_available():
        load_kwargs["max_memory"] = {0: args.gpu_memory, "cpu": args.cpu_memory}

    return load_kwargs


def sample_count_plan(args):
    ultrachat_samples = args.ultrachat_samples
    secondary_samples = args.secondary_samples

    if ultrachat_samples is None and secondary_samples is None:
        ultrachat_samples = args.num_calibration_samples // 2
        secondary_samples = args.num_calibration_samples - ultrachat_samples
    elif ultrachat_samples is None:
        ultrachat_samples = args.num_calibration_samples - secondary_samples
    elif secondary_samples is None:
        secondary_samples = args.num_calibration_samples - ultrachat_samples

    if ultrachat_samples <= 0 or secondary_samples <= 0:
        raise ValueError("mixed dataset split must allocate at least 1 sample per source")
    if ultrachat_samples + secondary_samples != args.num_calibration_samples:
        raise ValueError("ultrachat_samples + secondary_samples must equal num_calibration_samples")

    return ultrachat_samples, secondary_samples


def render_chat_sample(tokenizer, messages):
    filtered_messages = []
    for message in messages:
        content = message.get("content")
        role = message.get("role")
        if role is None or content is None:
            continue
        if isinstance(content, str) and not content and role == "system":
            continue
        filtered_messages.append({"role": role, "content": content})

    if not filtered_messages:
        raise ValueError("no usable messages after filtering")

    return tokenizer.apply_chat_template(
        filtered_messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_openplatypus_sample(tokenizer, row):
    prompt = row["instruction"]
    if row.get("input"):
        prompt = f"{prompt}\n\n{row['input']}"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["output"]},
    ]
    return render_chat_sample(tokenizer, messages)


def collect_stream_texts(stream, limit, render_fn):
    texts = []
    for row in stream:
        text = render_fn(row)
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def resolve_hf_token():
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        fallback_paths = [
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ]
        for path in fallback_paths:
            if path.exists():
                candidate = path.read_text().strip()
                if candidate:
                    token = candidate
                    break
    if not token:
        print("No HF token found; gated datasets may fail to load")
    return token


def build_calibration_dataset(args, tokenizer):
    hf_token = resolve_hf_token()

    if args.dataset_mode == "neuralmagic_llm":
        print(f"Loading calibration dataset {args.dataset}/{args.dataset_config} ...")
        return load_dataset(args.dataset, args.dataset_config, split="train", token=hf_token)

    ultrachat_samples, secondary_samples = sample_count_plan(args)

    print(
        "Building mixed calibration dataset "
        f"mode={args.dataset_mode} ultrachat={ultrachat_samples} secondary={secondary_samples} ..."
    )

    ultrachat_stream = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
        token=hf_token,
    ).shuffle(seed=42, buffer_size=10_000)
    ultrachat_texts = collect_stream_texts(
        ultrachat_stream,
        ultrachat_samples,
        lambda row: render_chat_sample(tokenizer, row["messages"]),
    )

    if len(ultrachat_texts) != ultrachat_samples:
        raise RuntimeError(
            f"Expected {ultrachat_samples} UltraChat samples, got {len(ultrachat_texts)}"
        )

    if args.dataset_mode == "ultrachat_nemotron":
        secondary_stream = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            split="chat",
            streaming=True,
            token=hf_token,
        ).shuffle(seed=43, buffer_size=10_000)
        secondary_texts = collect_stream_texts(
            secondary_stream,
            secondary_samples,
            lambda row: render_chat_sample(tokenizer, row["messages"]),
        )
    else:
        secondary_stream = load_dataset(
            "garage-bAInd/Open-Platypus",
            split="train",
            streaming=True,
            token=hf_token,
        ).shuffle(seed=43, buffer_size=10_000)
        secondary_texts = collect_stream_texts(
            secondary_stream,
            secondary_samples,
            lambda row: build_openplatypus_sample(tokenizer, row),
        )

    if len(secondary_texts) != secondary_samples:
        raise RuntimeError(
            f"Expected {secondary_samples} secondary samples, got {len(secondary_texts)}"
        )

    texts = ultrachat_texts + secondary_texts
    return Dataset.from_dict({"text": texts})


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    modeling_utils.caching_allocator_warmup = lambda *_, **__: None

    print(f"Loading source model from {args.model_dir} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir,
        **build_load_kwargs(args),
    )

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets=["Linear"],
        ignore=DEFAULT_IGNORE,
        scheme="NVFP4",
    )

    dataset = build_calibration_dataset(args, tokenizer)

    print(
        "Running oneshot NVFP4 quantization "
        f"({args.num_calibration_samples} samples, seq_len={args.max_seq_length}) ..."
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=dataset,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        text_column="text",
        save_compressed=False,
    )

    print("Calibration complete. Saving sharded quantized weights ...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    model.save_pretrained(
        args.output_dir,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(args.output_dir)
    for name in COPY_AFTER_SAVE:
        src = os.path.join(args.model_dir, name)
        dst = os.path.join(args.output_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    tokenizer_config_path = os.path.join(args.output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        import json
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
            tokenizer_config["tokenizer_class"] = "Qwen2TokenizerFast"
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
                f.write("\n")

    print(f"\nQuantization complete. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
