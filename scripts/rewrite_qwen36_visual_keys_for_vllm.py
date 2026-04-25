#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite model.visual.* keys to visual.* for vLLM-first artifact"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--extra-filename",
        default="model-multimodal-extra.safetensors",
        help="Extra shard filename that contains visual and MTP tensors",
    )
    return parser.parse_args()


def remap_key(key):
    if key.startswith("model.visual."):
        return key[len("model.") :]
    return key


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    extra_path = os.path.join(model_dir, args.extra_filename)

    index = json.load(open(index_path))
    if not os.path.exists(extra_path):
        raise FileNotFoundError(extra_path)

    remapped_tensors = {}
    with safe_open(extra_path, framework="pt") as handle:
        for key in handle.keys():
            remapped_tensors[remap_key(key)] = handle.get_tensor(key)

    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix="model-multimodal-extra-", suffix=".safetensors", dir=model_dir
    )
    os.close(tmp_fd)
    save_file(remapped_tensors, tmp_path)
    os.replace(tmp_path, extra_path)

    new_weight_map = {}
    for key, file_name in index["weight_map"].items():
        if file_name != args.extra_filename:
            new_weight_map[key] = file_name
            continue
        new_weight_map[remap_key(key)] = file_name

    with open(index_path, "w") as handle:
        json.dump(
            {"metadata": index.get("metadata", {}), "weight_map": dict(sorted(new_weight_map.items()))},
            handle,
            indent=2,
        )

    print(f"rewrote visual keys in {extra_path}")


if __name__ == "__main__":
    main()
