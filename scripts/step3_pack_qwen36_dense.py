#!/usr/bin/env python3
import argparse
import json
import os
import shutil

os.environ.setdefault("HF_HOME", os.path.abspath("./.hf_cache"))

from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_EXTRA_FILENAME = "model-multimodal-extra.safetensors"
COPY_IF_PRESENT = [
    "README.md",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix Qwen3.6 HauhauCS NVFP4 shard names and repack visual + MTP tensors"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Source HF checkpoint directory with visual and MTP tensors",
    )
    parser.add_argument(
        "--nvfp4-dir",
        required=True,
        help="Quantized output directory produced by llmcompressor",
    )
    parser.add_argument(
        "--shard-size",
        default="4GB",
        help="Maximum size of each main shard, for example 4GB",
    )
    parser.add_argument(
        "--extra-filename",
        default=DEFAULT_EXTRA_FILENAME,
        help="Filename used for preserved visual and MTP tensors",
    )
    return parser.parse_args()


def parse_shard_size(value):
    value = value.strip().upper()
    units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * multiplier)
    return int(value)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def save_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def build_extra_tensor_map(source_dir):
    source_index = load_json(os.path.join(source_dir, "model.safetensors.index.json"))
    extra_keys = sorted(
        key
        for key in source_index["weight_map"]
        if key.startswith("model.visual.") or key.startswith("mtp.")
    )
    print(f"Found {len(extra_keys)} extra tensors in source checkpoint")

    tensors = {}
    shard_to_keys = {}
    for key in extra_keys:
        shard_to_keys.setdefault(source_index["weight_map"][key], []).append(key)

    for shard_name, keys in sorted(shard_to_keys.items()):
        shard_path = os.path.join(source_dir, shard_name)
        with safe_open(shard_path, framework="pt") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
        print(f"  loaded {len(keys)} extra tensors from {shard_name}")

    return tensors


def collapse_language_model_prefix(key):
    needle = "model.language_model.language_model."
    while needle in key:
        key = key.replace(needle, "model.language_model.")
    return key


def remap_quantized_key(key):
    if key == "lm_head.weight" or key.startswith("model.visual.") or key.startswith("mtp."):
        return key

    key = collapse_language_model_prefix(key)
    if key.startswith("model.language_model."):
        return key
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


def should_skip_quantized_main_key(key):
    return (
        key.startswith("model.visual.")
        or key.startswith("model.language_model.visual.")
        or key.startswith("visual.")
        or key.startswith("mtp.")
    )


class TempShardWriter:
    def __init__(self, output_dir, max_bytes):
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.current_tensors = {}
        self.current_size = 0
        self.current_numel = 0
        self.temp_files = []
        self.total_size = 0
        self.total_numel = 0

    def add(self, name, tensor):
        tensor_bytes = tensor.nelement() * tensor.element_size()
        if self.current_tensors and self.current_size + tensor_bytes > self.max_bytes:
            self.flush()
        self.current_tensors[name] = tensor
        self.current_size += tensor_bytes
        self.current_numel += tensor.nelement()

    def flush(self):
        if not self.current_tensors:
            return
        shard_idx = len(self.temp_files) + 1
        temp_name = f"tmp-model-{shard_idx:05d}-of-XXXXX.safetensors"
        temp_path = os.path.join(self.output_dir, temp_name)
        save_file(self.current_tensors, temp_path)
        self.temp_files.append(
            {
                "temp_name": temp_name,
                "keys": list(self.current_tensors.keys()),
                "size": self.current_size,
                "numel": self.current_numel,
            }
        )
        self.total_size += self.current_size
        self.total_numel += self.current_numel
        print(
            f"  wrote temporary main shard {temp_name} "
            f"({len(self.current_tensors)} tensors, {self.current_size / 1e9:.2f} GB)"
        )
        self.current_tensors = {}
        self.current_size = 0
        self.current_numel = 0

    def finalize(self):
        self.flush()

        weight_map = {}
        main_files = []
        shard_count = len(self.temp_files)
        for shard_idx, shard_info in enumerate(self.temp_files, start=1):
            final_name = f"model-{shard_idx:05d}-of-{shard_count:05d}.safetensors"
            src = os.path.join(self.output_dir, shard_info["temp_name"])
            dst = os.path.join(self.output_dir, final_name)
            os.replace(src, dst)
            for key in shard_info["keys"]:
                weight_map[key] = final_name
            main_files.append(final_name)
        return weight_map, main_files, self.total_size, self.total_numel


def collect_existing_quantized_files(index_data, extra_filename):
    files = set(index_data["weight_map"].values())
    return sorted(name for name in files if name != extra_filename)


def delete_old_outputs(output_dir, file_names):
    for name in file_names:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def remap_ignore_entry(item):
    if item.startswith("re:"):
        return item
    return remap_quantized_key(item)


def merge_config(source_dir, nvfp4_dir):
    source_config = load_json(os.path.join(source_dir, "config.json"))
    quant_config = load_json(os.path.join(nvfp4_dir, "config.json"))

    merged = dict(source_config)
    for key in ("quantization_config", "dtype", "torch_dtype", "transformers_version"):
        if key in quant_config:
            merged[key] = quant_config[key]

    merged["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    merged.setdefault("dtype", merged.get("torch_dtype", "bfloat16"))

    quantization_config = merged.get("quantization_config")
    if quantization_config and "ignore" in quantization_config:
        remapped_ignore = [remap_ignore_entry(item) for item in quantization_config["ignore"]]

        vision_depth = source_config.get("vision_config", {}).get("depth", 0)
        for block_idx in range(vision_depth):
            prefix = f"model.visual.blocks.{block_idx}"
            remapped_ignore.extend(
                [
                    f"{prefix}.attn.qkv",
                    f"{prefix}.attn.proj",
                    f"{prefix}.mlp.linear_fc1",
                    f"{prefix}.mlp.linear_fc2",
                ]
            )
        remapped_ignore.extend(
            [
                "model.visual.merger.linear_fc1",
                "model.visual.merger.linear_fc2",
            ]
        )

        seen = set()
        deduped_ignore = []
        for item in remapped_ignore:
            if item not in seen:
                deduped_ignore.append(item)
                seen.add(item)
        quantization_config["ignore"] = deduped_ignore

    save_json(os.path.join(nvfp4_dir, "config.json"), merged)


def copy_auxiliary_files(source_dir, nvfp4_dir):
    for name in COPY_IF_PRESENT:
        src = os.path.join(source_dir, name)
        dst = os.path.join(nvfp4_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  copied {name}")


def sanitize_tokenizer_config(nvfp4_dir):
    path = os.path.join(nvfp4_dir, "tokenizer_config.json")
    if not os.path.exists(path):
        return
    tokenizer_config = load_json(path)
    if tokenizer_config.get("tokenizer_class") in (None, "TokenizersBackend"):
        tokenizer_config["tokenizer_class"] = "Qwen2TokenizerFast"
        save_json(path, tokenizer_config)
        print("  rewrote tokenizer_class to Qwen2TokenizerFast in tokenizer_config.json")


def main():
    args = parse_args()
    shard_size = parse_shard_size(args.shard_size)

    quant_index_path = os.path.join(args.nvfp4_dir, "model.safetensors.index.json")
    quant_index = load_json(quant_index_path)
    old_main_files = collect_existing_quantized_files(quant_index, args.extra_filename)
    print(f"Found {len(old_main_files)} existing quantized shards")

    writer = TempShardWriter(args.nvfp4_dir, shard_size)
    loaded_tensor_count = 0
    skipped_tensor_count = 0
    for shard_name in old_main_files:
        shard_path = os.path.join(args.nvfp4_dir, shard_name)
        with safe_open(shard_path, framework="pt") as handle:
            for key in sorted(handle.keys()):
                if should_skip_quantized_main_key(key):
                    skipped_tensor_count += 1
                    continue
                remapped_key = remap_quantized_key(key)
                writer.add(remapped_key, handle.get_tensor(key))
                loaded_tensor_count += 1
        print(f"  streamed main tensors from {shard_name}")

    extra_tensors = build_extra_tensor_map(args.source_dir)
    extra_path = os.path.join(args.nvfp4_dir, args.extra_filename)
    save_file(extra_tensors, extra_path)
    extra_size = sum(t.nelement() * t.element_size() for t in extra_tensors.values())
    extra_numel = sum(t.nelement() for t in extra_tensors.values())
    print(
        f"Saved {len(extra_tensors)} extra tensors to {args.extra_filename} "
        f"({extra_size / 1e9:.2f} GB)"
    )

    delete_old_outputs(args.nvfp4_dir, old_main_files)
    weight_map, new_main_files, main_size, main_numel = writer.finalize()
    for key in extra_tensors:
        weight_map[key] = args.extra_filename

    metadata = {
        "total_parameters": main_numel + extra_numel,
        "total_size": main_size + extra_size,
        "hybrid_extra_tensor_bytes": extra_size,
        "hybrid_extra_tensor_count": len(extra_tensors),
    }
    save_json(
        quant_index_path,
        {
            "metadata": metadata,
            "weight_map": dict(sorted(weight_map.items())),
        },
    )

    merge_config(args.source_dir, args.nvfp4_dir)
    copy_auxiliary_files(args.source_dir, args.nvfp4_dir)
    sanitize_tokenizer_config(args.nvfp4_dir)

    print(
        "\nPacking complete. "
        f"{loaded_tensor_count} quantized tensors kept in {len(new_main_files)} main shards, "
        f"{len(extra_tensors)} extra tensors saved separately, "
        f"{skipped_tensor_count} stray visual/MTP tensors dropped from quantized shards."
    )


if __name__ == "__main__":
    main()
