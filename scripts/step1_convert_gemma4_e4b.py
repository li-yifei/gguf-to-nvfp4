#!/usr/bin/env python3
"""step1_convert_gemma4_e4b.py -- Gemma 4 E4B GGUF -> HF safetensors.

Converts a text-model GGUF (e.g. HauhauCS's Q8_K_P) for a Gemma 4 E4B
multimodal model into HuggingFace safetensors format, ready for NVFP4
quantization via llmcompressor.

Strategy:
- Text tensors: dequantize from the text GGUF (handles F16/BF16/F32/Q8_0
  uniformly via gguf.quants.dequantize, which returns row-major f32)
- Vision / audio / embed_vision / embed_audio: copy directly from a
  reference HF repo (e.g. huihui-ai/Huihui-gemma-4-E4B-it-abliterated).
  This avoids having to synthesize the `.linear.weight` nesting and the
  FakeQuantize scalar bounds (input_max/input_min/output_max/output_min)
  that HF Gemma 4 expects on every vision/audio linear. The assumption
  is that finetunes don't modify vision/audio towers -- they inherit
  unchanged from google/gemma-4-e4b-it. Verify this before using the
  pipeline on a new finetune.

Gemma 4 RMSNorm caveats:
- Unlike Qwen3.5/Gemma 2/3, Gemma 4 does NOT use the `(1 + weight)`
  convention. Gemma4RMSNorm.forward is `normed_output * weight.float()`.
  GGUF stores the full weight value directly. Do NOT subtract 1.

Known tensor naming gotchas:
- GGUF `layer_output_scale.weight` -> HF `layer_scalar` (no `.weight`)
- GGUF `per_layer_token_embd.weight` -> HF `embed_tokens_per_layer.weight`
- No `lm_head.weight` in the HF state dict (tied to embed_tokens)
- GGUF `rope_freqs.weight` has no HF counterpart (runtime-computed)
"""

import argparse
import gc
import glob
import json
import os
import shutil

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

NUM_LAYERS = 42

GLOBAL_MAPPING = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output_norm.weight": "model.language_model.norm.weight",
    "per_layer_model_proj.weight": "model.language_model.per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "model.language_model.per_layer_projection_norm.weight",
    "per_layer_token_embd.weight": "model.language_model.embed_tokens_per_layer.weight",
}

LAYER_SUFFIX_MAPPING = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "post_norm.weight": "post_per_layer_input_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "inp_gate.weight": "per_layer_input_gate.weight",
    "proj.weight": "per_layer_projection.weight",
    "layer_output_scale.weight": "layer_scalar",  # no .weight suffix in HF
}

SKIP_TENSORS = {"rope_freqs.weight"}

CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
]


def build_text_mapping():
    mapping = dict(GLOBAL_MAPPING)
    for i in range(NUM_LAYERS):
        for gsfx, hsfx in LAYER_SUFFIX_MAPPING.items():
            mapping[f"blk.{i}.{gsfx}"] = f"model.language_model.layers.{i}.{hsfx}"
    return mapping


def parse_shard_size(s):
    s = s.strip().upper()
    mults = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suf, m in sorted(mults.items(), key=lambda x: -len(x[0])):
        if s.endswith(suf):
            return int(float(s[: -len(suf)]) * m)
    return int(s)


class ShardWriter:
    def __init__(self, output_dir, max_bytes):
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.current_shard = {}
        self.current_size = 0
        self.shard_idx = 0
        self.shard_files = []
        self.weight_map = {}
        self.total_size = 0

    def add(self, name, tensor):
        t_bytes = tensor.nelement() * tensor.element_size()
        if self.current_size + t_bytes > self.max_bytes and self.current_shard:
            self._flush()
        self.current_shard[name] = tensor
        self.current_size += t_bytes
        self.total_size += t_bytes

    def _flush(self):
        self.shard_idx += 1
        fname = f"model.safetensors-{self.shard_idx:05d}-of-XXXXX.safetensors"
        save_file(self.current_shard, os.path.join(self.output_dir, fname))
        self.shard_files.append((fname, list(self.current_shard.keys())))
        print(
            f"  Saved shard {self.shard_idx} "
            f"({len(self.current_shard)} tensors, {self.current_size / 1e9:.2f} GB)"
        )
        self.current_shard = {}
        self.current_size = 0
        gc.collect()

    def finalize(self):
        if self.current_shard:
            self._flush()
        n = len(self.shard_files)
        for i, (old, keys) in enumerate(self.shard_files):
            new = f"model.safetensors-{i+1:05d}-of-{n:05d}.safetensors"
            if old != new:
                os.rename(
                    os.path.join(self.output_dir, old),
                    os.path.join(self.output_dir, new),
                )
            for k in keys:
                self.weight_map[k] = new
        idx = {
            "metadata": {"total_size": self.total_size},
            "weight_map": dict(sorted(self.weight_map.items())),
        }
        with open(os.path.join(self.output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f, indent=2)
        return n


def gguf_tensor_to_bf16(tinfo):
    deq = dequantize(tinfo.data, tinfo.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(deq)).to(torch.bfloat16)


def copy_config_files(ref_snapshot, output_dir):
    for fname in CONFIG_FILES:
        src = os.path.join(ref_snapshot, fname)
        if not os.path.exists(src):
            continue
        real = os.path.realpath(src)  # resolve HF cache symlink
        shutil.copy2(real, os.path.join(output_dir, fname))
        print(f"  Copied {fname}")


def process_text_gguf(gguf_path, mapping, writer, skip_names):
    print(f"\nProcessing text GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)
    print(f"  {len(reader.tensors)} tensors in GGUF")

    unmapped = []
    converted = 0
    for tinfo in reader.tensors:
        gname = tinfo.name
        if gname in skip_names:
            print(f"  Skipped {gname}")
            continue
        if gname not in mapping:
            unmapped.append(gname)
            continue
        hname = mapping[gname]
        t = gguf_tensor_to_bf16(tinfo)
        writer.add(hname, t)
        converted += 1
        if converted % 100 == 0:
            print(f"  converted {converted} tensors...")
            gc.collect()

    print(f"  Text done: {converted} converted, {len(unmapped)} unmapped")
    for u in unmapped:
        print(f"    unmapped: {u}")
    del reader
    gc.collect()


def copy_nontext_from_reference(ref_snapshot, writer):
    print(f"\nCopying vision/audio/embed tensors from reference: {ref_snapshot}")
    st_files = sorted(glob.glob(os.path.join(ref_snapshot, "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No safetensors in {ref_snapshot}")

    def is_nontext(name):
        return (
            "vision_tower" in name
            or "audio_tower" in name
            or "embed_vision" in name
            or "embed_audio" in name
        )

    copied = 0
    for sf in st_files:
        with safe_open(sf, framework="pt") as f:
            for key in f.keys():
                if is_nontext(key):
                    writer.add(key, f.get_tensor(key))
                    copied += 1
    print(f"  Copied {copied} non-text tensors")


def verify_against_reference(ref_snapshot, our_keys):
    idx_path = os.path.join(ref_snapshot, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            ref_keys = set(json.load(f)["weight_map"].keys())
    else:
        ref_keys = set()
        for sf in sorted(glob.glob(os.path.join(ref_snapshot, "*.safetensors"))):
            with safe_open(sf, framework="pt") as f:
                ref_keys.update(f.keys())

    missing = ref_keys - our_keys
    extra = our_keys - ref_keys
    if missing:
        print(f"\n  MISSING {len(missing)} tensors vs reference:")
        for k in sorted(missing)[:30]:
            print(f"    {k}")
        if len(missing) > 30:
            print(f"    ... +{len(missing) - 30} more")
    if extra:
        print(f"\n  EXTRA {len(extra)} tensors (not in reference):")
        for k in sorted(extra)[:30]:
            print(f"    {k}")
    if not missing and not extra:
        print("\n  All tensor names match reference.")
    return len(missing), len(extra)


def main():
    p = argparse.ArgumentParser(
        description="Convert a Gemma 4 E4B text GGUF to HF safetensors format"
    )
    p.add_argument(
        "--gguf-text",
        required=True,
        help="Path to the text GGUF file (e.g. HauhauCS Q8_K_P)",
    )
    p.add_argument(
        "--reference-repo",
        default="huihui-ai/Huihui-gemma-4-E4B-it-abliterated",
        help="HF repo providing config.json, tokenizer, and vision/audio tensors. "
        "Must be a valid Gemma 4 E4B HF model that can be loaded with "
        "Gemma4ForConditionalGeneration.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", default="4GB")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fetching reference repo: {args.reference_repo}")
    ref_snapshot = snapshot_download(
        repo_id=args.reference_repo,
        allow_patterns=CONFIG_FILES + ["*.safetensors", "model.safetensors.index.json"],
    )
    print(f"Reference snapshot: {ref_snapshot}")

    print("\nCopying config/tokenizer files from reference...")
    copy_config_files(ref_snapshot, args.output_dir)

    mapping = build_text_mapping()
    shard_bytes = parse_shard_size(args.shard_size)
    writer = ShardWriter(args.output_dir, shard_bytes)

    process_text_gguf(args.gguf_text, mapping, writer, SKIP_TENSORS)
    copy_nontext_from_reference(ref_snapshot, writer)

    print("\nFinalizing shards...")
    n_shards = writer.finalize()

    print("\nVerifying against reference tensor set...")
    missing, extra = verify_against_reference(ref_snapshot, set(writer.weight_map.keys()))

    print(
        f"\nConversion complete: {len(writer.weight_map)} tensors in {n_shards} shards, "
        f"{writer.total_size / 1e9:.2f} GB"
    )
    if missing:
        print("Not verification-clean -- fix mapping before quantizing.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
