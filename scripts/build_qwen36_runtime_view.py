#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


COPY_IF_PRESENT = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "merges.txt",
    "vocab.json",
    "README.md",
    "recipe.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a runtime-compatible model view for vLLM profiles"
    )
    parser.add_argument("--source-dir", required=True, help="Published full model directory")
    parser.add_argument("--output-dir", required=True, help="Runtime view output directory")
    parser.add_argument(
        "--profile",
        default="full",
        choices=["full", "text", "no-vision", "no-mtp"],
        help="Which runtime profile to create",
    )
    parser.add_argument(
        "--extra-filename",
        default="model-runtime-extra.safetensors",
        help="Filename to use for remapped runtime-only extra tensors",
    )
    parser.add_argument(
        "--link-mode",
        default="hardlink",
        choices=["hardlink", "symlink", "copy"],
        help="How to materialize main shard files in the output directory",
    )
    return parser.parse_args()


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def remap_ignore_entry(entry):
    if entry.startswith("re:"):
        return entry
    return remap_key(entry)


def should_include(key, profile):
    is_visual = key.startswith("model.visual.") or key.startswith("visual.")
    is_mtp = key.startswith("mtp.")

    if profile == "full":
        return is_visual or is_mtp
    if profile == "text":
        return False
    if profile == "no-vision":
        return is_mtp
    if profile == "no-mtp":
        return is_visual
    raise ValueError(f"unsupported profile {profile}")


def remap_key(key):
    if key.startswith("model.language_model.visual."):
        return "model.visual." + key[len("model.language_model.visual.") :]
    if key.startswith("visual."):
        return "model." + key
    return key


def build_runtime_extra(source_dir, output_dir, source_index, profile, extra_filename):
    extra_keys = sorted(
        key
        for key, file_name in source_index["weight_map"].items()
        if file_name == "model-multimodal-extra.safetensors" and should_include(key, profile)
    )
    if not extra_keys:
        return {}, 0

    source_extra_path = os.path.join(source_dir, "model-multimodal-extra.safetensors")
    runtime_tensors = {}
    with safe_open(source_extra_path, framework="pt") as handle:
        for key in extra_keys:
            runtime_tensors[remap_key(key)] = handle.get_tensor(key)

    runtime_extra_path = os.path.join(output_dir, extra_filename)
    save_file(runtime_tensors, runtime_extra_path)
    size_bytes = runtime_extra_path and os.path.getsize(runtime_extra_path)
    print(f"wrote {len(runtime_tensors)} runtime extra tensors to {runtime_extra_path}")
    return {key: extra_filename for key in runtime_tensors}, size_bytes


def materialize_main_files(
    source_dir,
    output_dir,
    source_index,
    main_weight_map,
    files_to_link,
    link_mode,
):
    allowed_keys_by_file = {}
    for key, file_name in main_weight_map.items():
        allowed_keys_by_file.setdefault(file_name, set()).add(key)

    for file_name in files_to_link:
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(output_dir, file_name)
        allowed_keys = allowed_keys_by_file.get(file_name, set())

        with safe_open(src, framework="pt") as handle:
            actual_keys = list(handle.keys())
            needs_rewrite = set(actual_keys) != allowed_keys
            if needs_rewrite:
                filtered_tensors = {
                    key: handle.get_tensor(key) for key in actual_keys if key in allowed_keys
                }

        if os.path.lexists(dst):
            os.remove(dst)

        if needs_rewrite:
            save_file(filtered_tensors, dst)
            print(
                f"rewrote {file_name}: kept {len(filtered_tensors)} indexed tensors, "
                f"dropped {len(actual_keys) - len(filtered_tensors)} stray tensors"
            )
            continue

        if link_mode == "hardlink":
            os.link(src, dst)
        elif link_mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)


def copy_metadata_files(source_dir, output_dir):
    for file_name in COPY_IF_PRESENT:
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(output_dir, file_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        config = load_json(config_path)
        quant_config = config.get("quantization_config")
        if quant_config and "ignore" in quant_config:
            quant_config["ignore"] = [remap_ignore_entry(item) for item in quant_config["ignore"]]
            save_json(config_path, config)


def main():
    args = parse_args()
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    source_index = load_json(os.path.join(source_dir, "model.safetensors.index.json"))

    main_weight_map = {}
    main_files = set()
    for key, file_name in source_index["weight_map"].items():
        if file_name == "model-multimodal-extra.safetensors":
            continue
        if "visual" in key:
            continue
        main_weight_map[key] = file_name
        main_files.add(file_name)

    materialize_main_files(
        source_dir,
        output_dir,
        source_index,
        main_weight_map,
        sorted(main_files),
        args.link_mode,
    )
    copy_metadata_files(source_dir, output_dir)

    extra_weight_map, extra_size = build_runtime_extra(
        source_dir,
        output_dir,
        source_index,
        args.profile,
        args.extra_filename,
    )

    metadata = dict(source_index.get("metadata", {}))
    if extra_weight_map:
        metadata["runtime_extra_tensor_count"] = len(extra_weight_map)
        metadata["runtime_extra_tensor_bytes"] = extra_size
    else:
        metadata["runtime_extra_tensor_count"] = 0
        metadata["runtime_extra_tensor_bytes"] = 0

    weight_map = dict(sorted({**main_weight_map, **extra_weight_map}.items()))
    save_json(
        os.path.join(output_dir, "model.safetensors.index.json"),
        {"metadata": metadata, "weight_map": weight_map},
    )
    print(f"runtime view ready at {output_dir} profile={args.profile}")


if __name__ == "__main__":
    main()
