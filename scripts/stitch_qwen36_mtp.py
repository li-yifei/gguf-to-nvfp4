#!/usr/bin/env python3
import argparse
import json
import os
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy Qwen3.6 MTP tensors from an official HF shard directory into a converted dense checkpoint"
    )
    parser.add_argument("--source-dir", required=True, help="Official Qwen3.6 HF shard directory containing mtp.* tensors")
    parser.add_argument("--target-dir", required=True, help="Converted target HF checkpoint directory")
    parser.add_argument(
        "--extra-file",
        default="model-mtp-extra.safetensors",
        help="Extra safetensors filename to write in target dir",
    )
    return parser.parse_args()


args = parse_args()
SOURCE_DIR = args.source_dir
TARGET_DIR = args.target_dir
EXTRA_FILE = args.extra_file

source_index_path = os.path.join(SOURCE_DIR, "model.safetensors.index.json")
target_index_path = os.path.join(TARGET_DIR, "model.safetensors.index.json")
with open(source_index_path) as handle:
    source_index = json.load(handle)
with open(target_index_path) as handle:
    target_index = json.load(handle)

mtp_keys = sorted(key for key in source_index["weight_map"] if key.startswith("mtp."))
existing = [key for key in mtp_keys if key in target_index["weight_map"]]
if existing:
    print(f"Target already has {len(existing)} MTP tensors; skipping")
    raise SystemExit(0)

shard_to_keys = {}
for key in mtp_keys:
    shard_to_keys.setdefault(source_index["weight_map"][key], []).append(key)

tensors = {}
for shard_name, keys in sorted(shard_to_keys.items()):
    shard_path = os.path.join(SOURCE_DIR, shard_name)
    with safe_open(shard_path, framework="pt") as handle:
        for key in keys:
            tensors[key] = handle.get_tensor(key)
    print(f"loaded {len(keys)} MTP tensors from {shard_name}")

extra_path = os.path.join(TARGET_DIR, EXTRA_FILE)
save_file(tensors, extra_path)
extra_size = sum(t.nelement() * t.element_size() for t in tensors.values())
extra_numel = sum(t.nelement() for t in tensors.values())

for key in mtp_keys:
    target_index["weight_map"][key] = EXTRA_FILE
metadata = target_index.setdefault("metadata", {})
metadata["mtp_extra_tensor_count"] = len(mtp_keys)
metadata["mtp_extra_tensor_bytes"] = extra_size
metadata["total_size"] = int(metadata.get("total_size", 0)) + extra_size
metadata["total_parameters"] = int(metadata.get("total_parameters", 0)) + extra_numel
with open(target_index_path, "w") as handle:
    json.dump({"metadata": metadata, "weight_map": dict(sorted(target_index["weight_map"].items()))}, handle, indent=2)
print(f"wrote {len(mtp_keys)} MTP tensors to {EXTRA_FILE} ({extra_size / 1e9:.2f} GB)")
