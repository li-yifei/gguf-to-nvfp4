#!/usr/bin/env python3
"""Convert dense Qwen3.6-27B GGUF + mmproj GGUF to HF safetensors.

This is the dense 27B Qwen3.6 path used for
HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive. It streams tensors into
shards, applies the Qwen3.5/Qwen3.6 linear-attention fixes, and preserves the
visual tower in BF16/F16 form for later packing.
"""

import argparse
import os

import gc
import json
import numpy as np
import torch
from gguf import GGUFReader
try:
    from gguf.quants import dequantize
except ImportError:
    from gguf import dequantize
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
import shutil


def parse_shard_size(value):
    value = value.strip().upper()
    units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * multiplier)
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert dense Qwen3.6-27B GGUF + mmproj GGUF to HF safetensors"
    )
    parser.add_argument("--gguf-text", required=True, help="Text-model GGUF path")
    parser.add_argument("--gguf-vision", required=True, help="mmproj/vision GGUF path")
    parser.add_argument(
        "--output-dir",
        default="./Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-HF",
        help="Output HF checkpoint directory",
    )
    parser.add_argument(
        "--reference-repo",
        default="Qwen/Qwen3.6-27B",
        help="Reference HF repo for config/tokenizer/index metadata",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.abspath("./.hf_cache"),
        help="Hugging Face cache directory",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Optional HF_HOME override; defaults to --cache-dir if unset",
    )
    parser.add_argument(
        "--shard-size",
        default="4GB",
        help="Maximum shard size, e.g. 4GB",
    )
    return parser.parse_args()


ARGS = parse_args()
os.environ.setdefault("HF_HOME", ARGS.hf_home or ARGS.cache_dir)

GGUF_LLM_PATH = ARGS.gguf_text
GGUF_VISION_PATH = ARGS.gguf_vision
OUTPUT_DIR = ARGS.output_dir
REFERENCE_REPO = ARGS.reference_repo
CACHE_DIR = ARGS.cache_dir

FULL_ATTN_LAYERS = set(range(3, 64, 4))
NUM_VISION_BLOCKS = 27
SHARD_SIZE_BYTES = parse_shard_size(ARGS.shard_size)

GGUF_DTYPE_MAP = {
    0: torch.float32,
    1: torch.float16,
    30: torch.bfloat16,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: Download config/tokenizer ---
print("Downloading config and tokenizer from reference repo...")
config_files = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "merges.txt", "vocab.json",
    "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json",
]
for fname in config_files:
    dst = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(dst):
        print(f"  Already exists: {fname}")
        continue
    try:
        path = hf_hub_download(REFERENCE_REPO, fname, cache_dir=CACHE_DIR)
        shutil.copy2(path, dst)
        print(f"  Copied {fname}")
    except Exception as e:
        print(f"  Skipped {fname}: {e}")

# --- Step 2: Build mappings ---
def build_llm_mapping():
    mapping = {}
    mapping["token_embd.weight"] = "model.language_model.embed_tokens.weight"
    mapping["output.weight"] = "lm_head.weight"
    mapping["output_norm.weight"] = "model.language_model.norm.weight"
    for i in range(64):
        g, h = f"blk.{i}", f"model.language_model.layers.{i}"
        mapping[f"{g}.attn_norm.weight"] = f"{h}.input_layernorm.weight"
        mapping[f"{g}.post_attention_norm.weight"] = f"{h}.post_attention_layernorm.weight"
        mapping[f"{g}.ffn_down.weight"] = f"{h}.mlp.down_proj.weight"
        mapping[f"{g}.ffn_gate.weight"] = f"{h}.mlp.gate_proj.weight"
        mapping[f"{g}.ffn_up.weight"] = f"{h}.mlp.up_proj.weight"
        if i in FULL_ATTN_LAYERS:
            mapping[f"{g}.attn_q.weight"] = f"{h}.self_attn.q_proj.weight"
            mapping[f"{g}.attn_k.weight"] = f"{h}.self_attn.k_proj.weight"
            mapping[f"{g}.attn_v.weight"] = f"{h}.self_attn.v_proj.weight"
            mapping[f"{g}.attn_output.weight"] = f"{h}.self_attn.o_proj.weight"
            mapping[f"{g}.attn_q_norm.weight"] = f"{h}.self_attn.q_norm.weight"
            mapping[f"{g}.attn_k_norm.weight"] = f"{h}.self_attn.k_norm.weight"
        else:
            mapping[f"{g}.attn_qkv.weight"] = f"{h}.linear_attn.in_proj_qkv.weight"
            mapping[f"{g}.ssm_alpha.weight"] = f"{h}.linear_attn.in_proj_a.weight"
            mapping[f"{g}.ssm_beta.weight"] = f"{h}.linear_attn.in_proj_b.weight"
            mapping[f"{g}.attn_gate.weight"] = f"{h}.linear_attn.in_proj_z.weight"
            mapping[f"{g}.ssm_out.weight"] = f"{h}.linear_attn.out_proj.weight"
            mapping[f"{g}.ssm_a"] = f"{h}.linear_attn.A_log"
            mapping[f"{g}.ssm_conv1d.weight"] = f"{h}.linear_attn.conv1d.weight"
            mapping[f"{g}.ssm_dt.bias"] = f"{h}.linear_attn.dt_bias"
            mapping[f"{g}.ssm_norm.weight"] = f"{h}.linear_attn.norm.weight"
    return mapping

def build_vision_mapping():
    mapping = {}
    for i in range(NUM_VISION_BLOCKS):
        g, h = f"v.blk.{i}", f"model.visual.blocks.{i}"
        mapping[f"{g}.attn_qkv.weight"] = f"{h}.attn.qkv.weight"
        mapping[f"{g}.attn_qkv.bias"] = f"{h}.attn.qkv.bias"
        mapping[f"{g}.attn_out.weight"] = f"{h}.attn.proj.weight"
        mapping[f"{g}.attn_out.bias"] = f"{h}.attn.proj.bias"
        mapping[f"{g}.ffn_up.weight"] = f"{h}.mlp.linear_fc1.weight"
        mapping[f"{g}.ffn_up.bias"] = f"{h}.mlp.linear_fc1.bias"
        mapping[f"{g}.ffn_down.weight"] = f"{h}.mlp.linear_fc2.weight"
        mapping[f"{g}.ffn_down.bias"] = f"{h}.mlp.linear_fc2.bias"
        mapping[f"{g}.ln1.weight"] = f"{h}.norm1.weight"
        mapping[f"{g}.ln1.bias"] = f"{h}.norm1.bias"
        mapping[f"{g}.ln2.weight"] = f"{h}.norm2.weight"
        mapping[f"{g}.ln2.bias"] = f"{h}.norm2.bias"
    mapping["mm.0.weight"] = "model.visual.merger.linear_fc1.weight"
    mapping["mm.0.bias"] = "model.visual.merger.linear_fc1.bias"
    mapping["mm.2.weight"] = "model.visual.merger.linear_fc2.weight"
    mapping["mm.2.bias"] = "model.visual.merger.linear_fc2.bias"
    mapping["v.post_ln.weight"] = "model.visual.merger.norm.weight"
    mapping["v.post_ln.bias"] = "model.visual.merger.norm.bias"
    mapping["v.patch_embd.weight"] = "model.visual.patch_embed.proj.weight"
    mapping["v.patch_embd.bias"] = "model.visual.patch_embed.proj.bias"
    mapping["v.position_embd.weight"] = "model.visual.pos_embed.weight"
    return mapping

# --- Step 3: Tensor conversion ---
def gguf_tensor_to_torch(tensor_info, hf_name=""):
    dtype_val = tensor_info.tensor_type.value if hasattr(tensor_info.tensor_type, 'value') else tensor_info.tensor_type
    dtype = GGUF_DTYPE_MAP.get(dtype_val, torch.bfloat16)
    raw = tensor_info.data
    # GGUF stores shape in column-major order, reverse for row-major (PyTorch)
    shape = list(reversed(tensor_info.shape))

    if dtype_val not in GGUF_DTYPE_MAP:
        t = torch.from_numpy(dequantize(raw, tensor_info.tensor_type).copy()).to(torch.bfloat16)
    elif dtype in (torch.bfloat16, torch.float16):
        np_data = np.frombuffer(raw.tobytes(), dtype=np.uint16)
        t = torch.from_numpy(np_data.copy()).view(dtype).reshape(shape)
    else:
        np_data = np.frombuffer(raw.tobytes(), dtype=np.float32)
        t = torch.from_numpy(np_data.copy()).reshape(shape)

    # Special case: conv1d.weight [channels, kernel] -> [channels, 1, kernel]
    if "conv1d.weight" in hf_name and t.dim() == 2:
        t = t.unsqueeze(1).contiguous()  # [channels, 1, kernel_size]

    # GGUF stores RMSNorm weights as (1 + learned_param), HF expects learned_param
    _RMSNORM_GGUF_NAMES = ("attn_norm.", "post_attention_norm.", "output_norm.",
                           "attn_q_norm.", "attn_k_norm.")
    gguf_name = tensor_info.name
    if any(pat in gguf_name for pat in _RMSNORM_GGUF_NAMES):
        t = (t.float() - 1.0).to(torch.bfloat16)

    # GGUF stores A as -exp(A_log), HF expects A_log
    if gguf_name.endswith(".ssm_a"):
        t = (-t.float()).log().to(torch.bfloat16)

    # GGUF (llama.cpp) stores value heads in (3,16) grouping order,
    # HF expects (16,3) order. Unpermute all linear_attn tensors with 48 V-heads.
    V_HEADS, V_HEAD_DIM = 48, 128
    Q_K_SIZE = 4096  # Q(2048) + K(2048)
    V_SIZE = V_HEADS * V_HEAD_DIM  # 6144

    if "linear_attn" in hf_name:
        if hf_name.endswith((".A_log", ".dt_bias")) and t.shape[0] == V_HEADS:
            t = t.reshape(3, 16).permute(1, 0).contiguous().reshape(V_HEADS)
        elif hf_name.endswith((".in_proj_a.weight", ".in_proj_b.weight")):
            D = t.shape[1]
            t = t.reshape(3, 16, D).permute(1, 0, 2).contiguous().reshape(V_HEADS, D)
        elif hf_name.endswith(".in_proj_qkv.weight"):
            D = t.shape[1]
            QK, V = t[:Q_K_SIZE], t[Q_K_SIZE:]
            V = V.reshape(3, 16, V_HEAD_DIM, D).permute(1, 0, 2, 3).contiguous().reshape(V_SIZE, D)
            t = torch.cat([QK, V], dim=0)
        elif hf_name.endswith(".in_proj_z.weight"):
            D = t.shape[1]
            t = t.reshape(3, 16, V_HEAD_DIM, D).permute(1, 0, 2, 3).contiguous().reshape(V_SIZE, D)
        elif hf_name.endswith(".out_proj.weight"):
            D = t.shape[0]
            t = t.reshape(D, 3, 16, V_HEAD_DIM).permute(0, 2, 1, 3).contiguous().reshape(D, V_SIZE)
        elif "conv1d.weight" in hf_name and t.shape[0] > Q_K_SIZE:
            K = t.shape[2]
            QK, V = t[:Q_K_SIZE], t[Q_K_SIZE:]
            V = V.reshape(3, 16, V_HEAD_DIM, 1, K).permute(1, 0, 2, 3, 4).contiguous().reshape(V_SIZE, 1, K)
            t = torch.cat([QK, V], dim=0)

    return t

# --- Step 4: Streaming extract + shard save ---
class ShardWriter:
    def __init__(self, output_dir, max_bytes):
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.current_shard = {}
        self.current_size = 0
        self.shard_idx = 0
        self.shard_files = []  # list of (filename, [keys])
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
        # Use temp name, will rename at finalize
        fname = f"model.safetensors-{self.shard_idx:05d}-of-XXXXX.safetensors"
        fpath = os.path.join(self.output_dir, fname)
        save_file(self.current_shard, fpath)
        keys = list(self.current_shard.keys())
        self.shard_files.append((fname, keys))
        print(f"  Saved shard {self.shard_idx} ({len(keys)} tensors, {self.current_size / 1e9:.2f} GB)")
        self.current_shard = {}
        self.current_size = 0
        gc.collect()

    def finalize(self):
        if self.current_shard:
            self._flush()
        n = len(self.shard_files)
        for i, (old_name, keys) in enumerate(self.shard_files):
            new_name = f"model.safetensors-{i+1:05d}-of-{n:05d}.safetensors"
            old_path = os.path.join(self.output_dir, old_name)
            new_path = os.path.join(self.output_dir, new_name)
            if old_name != new_name:
                os.rename(old_path, new_path)
            for k in keys:
                self.weight_map[k] = new_name

        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": dict(sorted(self.weight_map.items()))
        }
        with open(os.path.join(self.output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        return n

def process_gguf(gguf_path, name_mapping, label, writer):
    print(f"\nProcessing {label}: {gguf_path}")
    reader = GGUFReader(gguf_path)
    print(f"  {len(reader.tensors)} tensors in GGUF")

    converted = 0
    unmapped = []
    for tensor_info in reader.tensors:
        gguf_name = tensor_info.name
        if gguf_name not in name_mapping:
            if gguf_name == "v.patch_embd.weight.1":
                print(f"  NOTE: Skipping {gguf_name} (temporal patch embed duplicate)")
                continue
            unmapped.append(gguf_name)
            continue

        hf_name = name_mapping[gguf_name]
        t = gguf_tensor_to_torch(tensor_info, hf_name)
        writer.add(hf_name, t)
        converted += 1
        if converted % 50 == 0:
            print(f"  Converted {converted} tensors...")
            gc.collect()

    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped tensors:")
        for name in unmapped:
            print(f"    {name}")

    print(f"  Done: {converted} tensors converted")
    # Free the reader
    del reader
    gc.collect()

# --- Main ---
llm_mapping = build_llm_mapping()
vision_mapping = build_vision_mapping()

writer = ShardWriter(OUTPUT_DIR, SHARD_SIZE_BYTES)

process_gguf(GGUF_LLM_PATH, llm_mapping, "LLM", writer)
process_gguf(GGUF_VISION_PATH, vision_mapping, "Vision", writer)

print("\nFinalizing shards...")
n_shards = writer.finalize()

# --- Verify ---
print("\nVerifying against reference model index...")
idx_path = hf_hub_download(REFERENCE_REPO, "model.safetensors.index.json", cache_dir=CACHE_DIR)
with open(idx_path) as f:
    ref_index = json.load(f)
ref_keys = set(ref_index["weight_map"].keys())
our_keys = set(writer.weight_map.keys())

missing = ref_keys - our_keys
extra = our_keys - ref_keys
if missing:
    print(f"  Missing {len(missing)} tensors vs reference:")
    for k in sorted(missing):
        print(f"    {k}")
if extra:
    print(f"  Extra {len(extra)} tensors (not in reference):")
    for k in sorted(extra):
        print(f"    {k}")
if not missing and not extra:
    print("  All tensor names match reference!")

print(f"\nConversion complete! {len(our_keys)} tensors in {n_shards} shards")
print(f"Total size: {writer.total_size / 1e9:.2f} GB")
