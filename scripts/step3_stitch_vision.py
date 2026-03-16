#!/usr/bin/env python3
# step3_stitch_vision.py -- Stitch vision weights back into NVFP4 quantized model
# The quantization only processes the text model (AutoModelForCausalLM).
# This script merges the original bf16 vision weights back and updates config
# to produce a complete Qwen3_5ForConditionalGeneration model.
import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def parse_shard_size(s: str) -> int:
    """Parse human-readable shard size (e.g. '4GB') to bytes."""
    s = s.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def main():
    parser = argparse.ArgumentParser(
        description="Stitch bf16 vision weights into an NVFP4 quantized model"
    )
    parser.add_argument("--bf16-dir", required=True,
                        help="bf16 model directory containing vision weights")
    parser.add_argument("--nvfp4-dir", required=True,
                        help="NVFP4 quantized model directory (modified in-place)")
    parser.add_argument("--shard-size", default="4GB",
                        help="Max shard size (e.g. '4GB', '2GB')")
    args = parser.parse_args()

    bf16_dir = args.bf16_dir
    final_dir = args.nvfp4_dir
    shard_size_bytes = parse_shard_size(args.shard_size)

    print("Step 3: Stitching vision weights into NVFP4 model...")

    # --- 1. Collect vision tensors from bf16 model ---
    print("Loading vision tensors from bf16 model...")
    bf16_index_path = os.path.join(bf16_dir, "model.safetensors.index.json")
    with open(bf16_index_path) as f:
        bf16_index = json.load(f)

    vision_keys = [k for k in bf16_index["weight_map"] if "visual" in k]
    print(f"  Found {len(vision_keys)} vision tensors")

    # Load vision tensors
    vision_tensors = {}
    # Group by shard file for efficient loading
    shard_to_keys = {}
    for key in vision_keys:
        shard = bf16_index["weight_map"][key]
        shard_to_keys.setdefault(shard, []).append(key)

    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(bf16_dir, shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                vision_tensors[key] = f.get_tensor(key)
        print(f"  Loaded {len(keys)} tensors from {shard_file}")

    # --- 2. Check if NVFP4 model already has vision tensors ---
    nvfp4_index_path = os.path.join(final_dir, "model.safetensors.index.json")
    if os.path.exists(nvfp4_index_path):
        with open(nvfp4_index_path) as f:
            nvfp4_index = json.load(f)
        existing_vision = [k for k in nvfp4_index["weight_map"] if "visual" in k]
        if existing_vision:
            print(f"  NVFP4 model already has {len(existing_vision)} vision tensors, skipping stitch")
        else:
            print("  NVFP4 model has no vision tensors, will add them")
    else:
        # Single model.safetensors file
        nvfp4_index = None

    # --- 3. Load all NVFP4 tensors and merge with vision ---
    print("Loading NVFP4 quantized tensors...")
    all_tensors = {}

    if nvfp4_index:
        # Sharded model
        shard_files = set(nvfp4_index["weight_map"].values())
        for shard_file in sorted(shard_files):
            shard_path = os.path.join(final_dir, shard_file)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    all_tensors[key] = f.get_tensor(key)
            print(f"  Loaded from {shard_file}")
    else:
        # Single file
        single_path = os.path.join(final_dir, "model.safetensors")
        if os.path.exists(single_path):
            with safe_open(single_path, framework="pt") as f:
                for key in f.keys():
                    all_tensors[key] = f.get_tensor(key)
            print(f"  Loaded from model.safetensors")

    # Remap: quantized model uses model.X but we need model.language_model.X
    # Check if remapping is needed
    needs_remap = any(k.startswith("model.layers.") for k in all_tensors)
    if needs_remap:
        print("  Remapping model.* -> model.language_model.*")
        remapped = {}
        for key, tensor in all_tensors.items():
            if key == "lm_head.weight":
                remapped[key] = tensor
            elif key.startswith("model."):
                new_key = "model.language_model." + key[len("model."):]
                remapped[new_key] = tensor
            else:
                remapped[key] = tensor
        all_tensors = remapped

    # Merge vision tensors
    print(f"Merging {len(vision_tensors)} vision tensors...")
    all_tensors.update(vision_tensors)
    print(f"Total tensors after merge: {len(all_tensors)}")

    # --- 4. Re-save as sharded safetensors ---
    print("Saving merged model with sharding...")
    # Remove old shard files
    for f_name in os.listdir(final_dir):
        if f_name.startswith("model.safetensors"):
            os.remove(os.path.join(final_dir, f_name))

    sorted_keys = sorted(all_tensors.keys())
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted_keys:
        t = all_tensors[key]
        t_bytes = t.nelement() * t.element_size()
        if current_size + t_bytes > shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = t
        current_size += t_bytes

    if current_shard:
        shards.append(current_shard)

    n_shards = len(shards)
    weight_map = {}
    total_size = 0

    for i, shard_data in enumerate(shards):
        fname = f"model.safetensors-{i+1:05d}-of-{n_shards:05d}.safetensors"
        fpath = os.path.join(final_dir, fname)
        save_file(shard_data, fpath)
        for key in shard_data:
            weight_map[key] = fname
            t = shard_data[key]
            total_size += t.nelement() * t.element_size()
        print(f"  Saved {fname} ({len(shard_data)} tensors)")

    # Save index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items()))
    }
    with open(os.path.join(final_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # --- 5. Update config.json to use Qwen3_5ForConditionalGeneration ---
    print("Updating config.json...")
    config_path = os.path.join(final_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Ensure architecture is set for multimodal
    config["architectures"] = ["Qwen3_5ForConditionalGeneration"]

    # Copy vision-related configs from bf16 if missing
    bf16_config_path = os.path.join(bf16_dir, "config.json")
    with open(bf16_config_path) as f:
        bf16_config = json.load(f)

    for key in ["vision_config", "vision_start_token_id", "vision_end_token_id",
                "image_token_id", "video_token_id"]:
        if key in bf16_config:
            config[key] = bf16_config[key]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Copy preprocessor configs if missing
    for fname in ["preprocessor_config.json", "video_preprocessor_config.json"]:
        src = os.path.join(bf16_dir, fname)
        dst = os.path.join(final_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    print(f"\nStitching complete! {len(all_tensors)} tensors in {n_shards} shards")
    print(f"Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
