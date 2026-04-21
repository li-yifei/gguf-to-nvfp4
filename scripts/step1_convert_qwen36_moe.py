#!/usr/bin/env python3
"""step1_convert_qwen36_moe.py -- Qwen3.6-35B-A3B MoE GGUF -> HF safetensors.

Converts a Q8_K_P GGUF for the Qwen3.6-35B-A3B MoE model into HF safetensors
format, ready for NVFP4 quantization via llmcompressor.

Strategy:
- Text tensors: dequantize from GGUF via gguf.quants.dequantize (handles
  F16/F32/Q8_K uniformly, returns row-major f32 with correct shape).
- MoE experts: GGUF stores gate_exps and up_exps separately; HF expects
  fused gate_up_proj. Concatenate along dim 1 after dequantize.
- Vision: dequantize from mmproj GGUF (same as Qwen3.5).
- MTP: copy from Qwen/Qwen3.6-35B-A3B reference (not in GGUF).
- Config/tokenizer: copy from reference HF repo.

Qwen3.6 vs Qwen3.5 key differences:
- MoE (256 experts, 8 routed + 1 shared) vs dense FFN
- V_HEADS=32, HEADS_PER_GROUP=2 vs 48, 3
- Q_K_SIZE=4096, V_SIZE=4096 vs 4096, 6144
- Full attn Q includes output gate (8192 wide, not 4096)
- 40 layers vs 64
- ssm_norm shape [128] (V_HEAD_DIM) vs [48] (V_HEADS)
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

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 40
NUM_VISION_BLOCKS = 27
FULL_ATTN_LAYERS = set(range(3, NUM_LAYERS, 4))  # 3,7,11,...,39

# Linear attention dimensions
V_HEADS = 32          # linear_num_value_heads
K_HEADS = 16          # linear_num_key_heads
HEADS_PER_GROUP = V_HEADS // K_HEADS  # 2
V_HEAD_DIM = 128      # linear_value_head_dim
V_SIZE = V_HEADS * V_HEAD_DIM  # 4096
Q_K_SIZE = 4096       # Q(2048) + K(2048)

# RMSNorm GGUF names that store (1 + learned_param)
_RMSNORM_GGUF_NAMES = (
    "attn_norm.", "post_attention_norm.", "output_norm.",
    "attn_q_norm.", "attn_k_norm.",
)

CONFIG_FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "merges.txt", "vocab.json",
    "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json",
]


# ---------------------------------------------------------------------------
# Tensor name mappings
# ---------------------------------------------------------------------------

def build_llm_mapping():
    """Build GGUF->HF name mapping for the text model."""
    m = {}
    m["token_embd.weight"] = "model.language_model.embed_tokens.weight"
    m["output.weight"] = "lm_head.weight"
    m["output_norm.weight"] = "model.language_model.norm.weight"

    for i in range(NUM_LAYERS):
        g = f"blk.{i}"
        h = f"model.language_model.layers.{i}"

        # Norms (all layers)
        m[f"{g}.attn_norm.weight"] = f"{h}.input_layernorm.weight"
        m[f"{g}.post_attention_norm.weight"] = f"{h}.post_attention_layernorm.weight"

        # MoE FFN (all layers)
        # gate_exps and up_exps use sentinel suffixes for deferred fusion
        m[f"{g}.ffn_gate_exps.weight"] = f"{h}.mlp.experts.gate_up_proj__gate"
        m[f"{g}.ffn_up_exps.weight"] = f"{h}.mlp.experts.gate_up_proj__up"
        m[f"{g}.ffn_down_exps.weight"] = f"{h}.mlp.experts.down_proj"
        m[f"{g}.ffn_gate_inp.weight"] = f"{h}.mlp.gate.weight"
        m[f"{g}.ffn_gate_shexp.weight"] = f"{h}.mlp.shared_expert.gate_proj.weight"
        m[f"{g}.ffn_up_shexp.weight"] = f"{h}.mlp.shared_expert.up_proj.weight"
        m[f"{g}.ffn_down_shexp.weight"] = f"{h}.mlp.shared_expert.down_proj.weight"
        m[f"{g}.ffn_gate_inp_shexp.weight"] = f"{h}.mlp.shared_expert_gate.weight"

        if i in FULL_ATTN_LAYERS:
            # Full (gated) attention
            m[f"{g}.attn_q.weight"] = f"{h}.self_attn.q_proj.weight"
            m[f"{g}.attn_k.weight"] = f"{h}.self_attn.k_proj.weight"
            m[f"{g}.attn_v.weight"] = f"{h}.self_attn.v_proj.weight"
            m[f"{g}.attn_output.weight"] = f"{h}.self_attn.o_proj.weight"
            m[f"{g}.attn_q_norm.weight"] = f"{h}.self_attn.q_norm.weight"
            m[f"{g}.attn_k_norm.weight"] = f"{h}.self_attn.k_norm.weight"
        else:
            # Linear attention (Gated DeltaNet)
            m[f"{g}.attn_qkv.weight"] = f"{h}.linear_attn.in_proj_qkv.weight"
            m[f"{g}.ssm_alpha.weight"] = f"{h}.linear_attn.in_proj_a.weight"
            m[f"{g}.ssm_beta.weight"] = f"{h}.linear_attn.in_proj_b.weight"
            m[f"{g}.attn_gate.weight"] = f"{h}.linear_attn.in_proj_z.weight"
            m[f"{g}.ssm_out.weight"] = f"{h}.linear_attn.out_proj.weight"
            m[f"{g}.ssm_a"] = f"{h}.linear_attn.A_log"
            m[f"{g}.ssm_conv1d.weight"] = f"{h}.linear_attn.conv1d.weight"
            m[f"{g}.ssm_dt.bias"] = f"{h}.linear_attn.dt_bias"
            m[f"{g}.ssm_norm.weight"] = f"{h}.linear_attn.norm.weight"

    return m


def build_vision_mapping():
    """Build GGUF->HF name mapping for the vision encoder (mmproj)."""
    m = {}
    for i in range(NUM_VISION_BLOCKS):
        g, h = f"v.blk.{i}", f"model.visual.blocks.{i}"
        m[f"{g}.attn_qkv.weight"] = f"{h}.attn.qkv.weight"
        m[f"{g}.attn_qkv.bias"] = f"{h}.attn.qkv.bias"
        m[f"{g}.attn_out.weight"] = f"{h}.attn.proj.weight"
        m[f"{g}.attn_out.bias"] = f"{h}.attn.proj.bias"
        m[f"{g}.ffn_up.weight"] = f"{h}.mlp.linear_fc1.weight"
        m[f"{g}.ffn_up.bias"] = f"{h}.mlp.linear_fc1.bias"
        m[f"{g}.ffn_down.weight"] = f"{h}.mlp.linear_fc2.weight"
        m[f"{g}.ffn_down.bias"] = f"{h}.mlp.linear_fc2.bias"
        m[f"{g}.ln1.weight"] = f"{h}.norm1.weight"
        m[f"{g}.ln1.bias"] = f"{h}.norm1.bias"
        m[f"{g}.ln2.weight"] = f"{h}.norm2.weight"
        m[f"{g}.ln2.bias"] = f"{h}.norm2.bias"
    m["mm.0.weight"] = "model.visual.merger.linear_fc1.weight"
    m["mm.0.bias"] = "model.visual.merger.linear_fc1.bias"
    m["mm.2.weight"] = "model.visual.merger.linear_fc2.weight"
    m["mm.2.bias"] = "model.visual.merger.linear_fc2.bias"
    m["v.post_ln.weight"] = "model.visual.merger.norm.weight"
    m["v.post_ln.bias"] = "model.visual.merger.norm.bias"
    # patch_embd: .weight and .weight.1 are stacked into 5D [C,3,T,H,W]
    m["v.patch_embd.weight"] = "model.visual.patch_embed.proj.weight__t0"
    m["v.patch_embd.weight.1"] = "model.visual.patch_embed.proj.weight__t1"
    m["v.patch_embd.bias"] = "model.visual.patch_embed.proj.bias"
    m["v.position_embd.weight"] = "model.visual.pos_embed.weight"
    return m


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

def gguf_tensor_to_bf16(tinfo):
    """Dequantize any GGUF tensor to bf16. Handles F32/F16/Q8_K uniformly."""
    deq = dequantize(tinfo.data, tinfo.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(deq)).to(torch.bfloat16)


def apply_fixes(t, gguf_name, hf_name):
    """Apply domain/shape fixes for specific tensor types."""

    # Conv1d: unsqueeze [channels, kernel] -> [channels, 1, kernel]
    if "conv1d.weight" in hf_name and t.dim() == 2:
        t = t.unsqueeze(1).contiguous()

    # shared_expert_gate: model expects [1, hidden] not [hidden]
    if hf_name.endswith(".shared_expert_gate.weight") and t.dim() == 1:
        t = t.unsqueeze(0).contiguous()

    # RMSNorm: GGUF stores (1 + weight), HF expects weight
    if any(pat in gguf_name for pat in _RMSNORM_GGUF_NAMES):
        t = (t.float() - 1.0).to(torch.bfloat16)

    # A_log: GGUF stores A = -exp(A_log), HF expects A_log
    if gguf_name.endswith(".ssm_a"):
        t = (-t.float()).log().to(torch.bfloat16)

    # Linear attention V-head permutation: (HEADS_PER_GROUP, K_HEADS) -> (K_HEADS, HEADS_PER_GROUP)
    if "linear_attn" in hf_name:
        t = _permute_v_heads(t, hf_name, gguf_name)

    return t


def _permute_v_heads(t, hf_name, gguf_name):
    """Permute V-head ordering from GGUF (2,16) to HF (16,2)."""
    HPG, KH = HEADS_PER_GROUP, K_HEADS  # 2, 16

    if hf_name.endswith((".A_log", ".dt_bias")) and t.shape[0] == V_HEADS:
        return t.reshape(HPG, KH).permute(1, 0).contiguous().reshape(V_HEADS)

    if hf_name.endswith((".in_proj_a.weight", ".in_proj_b.weight")):
        D = t.shape[1]
        return t.reshape(HPG, KH, D).permute(1, 0, 2).contiguous().reshape(V_HEADS, D)

    if hf_name.endswith(".in_proj_qkv.weight"):
        D = t.shape[1]
        QK, V = t[:Q_K_SIZE], t[Q_K_SIZE:]
        V = V.reshape(HPG, KH, V_HEAD_DIM, D).permute(1, 0, 2, 3).contiguous().reshape(V_SIZE, D)
        return torch.cat([QK, V], dim=0)

    if hf_name.endswith(".in_proj_z.weight"):
        D = t.shape[1]
        return t.reshape(HPG, KH, V_HEAD_DIM, D).permute(1, 0, 2, 3).contiguous().reshape(V_SIZE, D)

    if hf_name.endswith(".out_proj.weight"):
        D = t.shape[0]
        return t.reshape(D, HPG, KH, V_HEAD_DIM).permute(0, 2, 1, 3).contiguous().reshape(D, V_SIZE)

    if "conv1d.weight" in hf_name and t.shape[0] > Q_K_SIZE:
        K = t.shape[2]
        QK, V = t[:Q_K_SIZE], t[Q_K_SIZE:]
        V = V.reshape(HPG, KH, V_HEAD_DIM, 1, K).permute(1, 0, 2, 3, 4).contiguous().reshape(V_SIZE, 1, K)
        return torch.cat([QK, V], dim=0)

    return t


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

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
        print(f"  Saved shard {self.shard_idx} "
              f"({len(self.current_shard)} tensors, {self.current_size / 1e9:.2f} GB)")
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


# ---------------------------------------------------------------------------
# GGUF processing
# ---------------------------------------------------------------------------

def process_text_gguf(gguf_path, mapping, writer):
    """Extract and convert text tensors from GGUF, handling MoE expert fusion."""
    print(f"\nProcessing text GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)
    print(f"  {len(reader.tensors)} tensors in GGUF")

    # Collect gate_exps and up_exps for deferred fusion
    pending_gate_exps = {}  # layer_prefix -> tensor
    pending_up_exps = {}

    converted, unmapped_list = 0, []
    for tinfo in reader.tensors:
        gname = tinfo.name
        if gname not in mapping:
            unmapped_list.append(gname)
            continue

        hname = mapping[gname]
        t = gguf_tensor_to_bf16(tinfo)
        t = apply_fixes(t, gname, hname)

        # Handle expert gate/up fusion
        if hname.endswith("gate_up_proj__gate"):
            layer_key = hname.rsplit(".", 1)[0]  # ...mlp.experts
            pending_gate_exps[layer_key] = t
            if layer_key in pending_up_exps:
                fused = torch.cat([pending_gate_exps.pop(layer_key),
                                   pending_up_exps.pop(layer_key)], dim=1)
                writer.add(f"{layer_key}.gate_up_proj", fused)
                converted += 1
                del fused
            continue

        if hname.endswith("gate_up_proj__up"):
            layer_key = hname.rsplit(".", 1)[0]
            pending_up_exps[layer_key] = t
            if layer_key in pending_gate_exps:
                fused = torch.cat([pending_gate_exps.pop(layer_key),
                                   pending_up_exps.pop(layer_key)], dim=1)
                writer.add(f"{layer_key}.gate_up_proj", fused)
                converted += 1
                del fused
            continue

        writer.add(hname, t)
        converted += 1
        if converted % 50 == 0:
            print(f"  Converted {converted} tensors...")
            gc.collect()

    # Flush any unpaired experts (should not happen)
    if pending_gate_exps or pending_up_exps:
        print(f"  WARNING: {len(pending_gate_exps)} unpaired gate_exps, "
              f"{len(pending_up_exps)} unpaired up_exps")

    if unmapped_list:
        print(f"  WARNING: {len(unmapped_list)} unmapped GGUF tensors:")
        for n in unmapped_list:
            print(f"    {n}")

    print(f"  Text done: {converted} tensors converted")
    del reader
    gc.collect()


def process_vision_gguf(gguf_path, mapping, writer):
    """Extract vision tensors from mmproj GGUF."""
    print(f"\nProcessing vision GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)
    print(f"  {len(reader.tensors)} tensors in GGUF")

    # Deferred patch_embed stacking (temporal dim)
    pending_patch_t0 = None
    pending_patch_t1 = None

    converted, unmapped_list = 0, []
    for tinfo in reader.tensors:
        gname = tinfo.name
        if gname not in mapping:
            unmapped_list.append(gname)
            continue

        hname = mapping[gname]
        t = gguf_tensor_to_bf16(tinfo)

        # Handle temporal patch embed stacking
        if hname.endswith("proj.weight__t0"):
            pending_patch_t0 = t
            if pending_patch_t1 is not None:
                # Stack [C,3,H,W] x2 -> [C,3,2,H,W]
                stacked = torch.stack([pending_patch_t0, pending_patch_t1], dim=2)
                writer.add("model.visual.patch_embed.proj.weight", stacked)
                converted += 1
                pending_patch_t0 = pending_patch_t1 = None
            continue

        if hname.endswith("proj.weight__t1"):
            pending_patch_t1 = t
            if pending_patch_t0 is not None:
                stacked = torch.stack([pending_patch_t0, pending_patch_t1], dim=2)
                writer.add("model.visual.patch_embed.proj.weight", stacked)
                converted += 1
                pending_patch_t0 = pending_patch_t1 = None
            continue

        writer.add(hname, t)
        converted += 1

    # Handle case where only one temporal frame exists (fallback)
    if pending_patch_t0 is not None and pending_patch_t1 is None:
        print("  WARNING: Only patch_embd.weight found, no .weight.1 — using 4D")
        writer.add("model.visual.patch_embed.proj.weight", pending_patch_t0)
        converted += 1

    if unmapped_list:
        print(f"  WARNING: {len(unmapped_list)} unmapped vision tensors:")
        for n in unmapped_list:
            print(f"    {n}")

    print(f"  Vision done: {converted} tensors converted")
    del reader
    gc.collect()


def copy_mtp_from_reference(reference_repo, writer):
    """Copy MTP (Multi-Token Prediction) tensors from reference HF model.

    Downloads only the index + specific shards containing mtp.* tensors,
    NOT the entire ~67GB model.
    """
    from huggingface_hub import hf_hub_download

    print(f"\nCopying MTP tensors from reference: {reference_repo}")

    # Step 1: download just the index to find which shards have MTP
    idx_file = hf_hub_download(reference_repo, "model.safetensors.index.json")
    with open(idx_file) as f:
        idx = json.load(f)

    mtp_shards = set()
    for key, shard in idx["weight_map"].items():
        if key.startswith("mtp."):
            mtp_shards.add(shard)

    if not mtp_shards:
        print("  WARNING: No MTP tensors found in reference model index")
        return 0

    print(f"  MTP tensors span {len(mtp_shards)} shard(s): {sorted(mtp_shards)}")

    # Step 2: download only those specific shards
    copied = 0
    for shard_name in sorted(mtp_shards):
        shard_path = hf_hub_download(reference_repo, shard_name)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    writer.add(key, f.get_tensor(key))
                    copied += 1
        print(f"  Loaded {shard_name}: {copied} MTP tensors so far")

    print(f"  Copied {copied} MTP tensors total")
    return copied


def copy_config_files(ref_snapshot, output_dir):
    """Copy config/tokenizer files from reference repo snapshot."""
    for fname in CONFIG_FILES:
        src = os.path.join(ref_snapshot, fname)
        if not os.path.exists(src):
            continue
        real = os.path.realpath(src)  # resolve HF cache symlink
        shutil.copy2(real, os.path.join(output_dir, fname))
        print(f"  Copied {fname}")


def verify_against_reference(ref_snapshot, our_keys):
    """Compare our tensor names against reference model."""
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
        for k in sorted(missing)[:40]:
            print(f"    {k}")
        if len(missing) > 40:
            print(f"    ... +{len(missing) - 40} more")
    if extra:
        print(f"\n  EXTRA {len(extra)} tensors (not in reference):")
        for k in sorted(extra)[:40]:
            print(f"    {k}")
    if not missing and not extra:
        print("\n  All tensor names match reference!")
    return len(missing), len(extra)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_shard_size(s):
    s = s.strip().upper()
    mults = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suf, m in sorted(mults.items(), key=lambda x: -len(x[0])):
        if s.endswith(suf):
            return int(float(s[:-len(suf)]) * m)
    return int(s)


def main():
    p = argparse.ArgumentParser(
        description="Convert Qwen3.6-35B-A3B MoE GGUF to HF safetensors"
    )
    p.add_argument("--gguf-llm", required=True,
                   help="Path to Q8_K_P LLM GGUF file")
    p.add_argument("--gguf-vision", default=None,
                   help="Path to mmproj GGUF file")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for HF safetensors")
    p.add_argument("--reference-repo", default="Qwen/Qwen3.6-35B-A3B",
                   help="HF repo for config/tokenizer/MTP")
    p.add_argument("--shard-size", default="4GB",
                   help="Max shard size")
    p.add_argument("--skip-mtp", action="store_true",
                   help="Skip copying MTP tensors from reference")
    args = p.parse_args()

    shard_bytes = _parse_shard_size(args.shard_size)
    os.makedirs(args.output_dir, exist_ok=True)

    # Download config/tokenizer from reference (lightweight, no safetensors)
    print(f"Fetching config/tokenizer from: {args.reference_repo}")
    ref_snapshot = snapshot_download(
        repo_id=args.reference_repo,
        allow_patterns=list(CONFIG_FILES) + ["model.safetensors.index.json"],
    )
    print(f"Reference snapshot: {ref_snapshot}")

    print("\nCopying config/tokenizer files...")
    copy_config_files(ref_snapshot, args.output_dir)

    writer = ShardWriter(args.output_dir, shard_bytes)

    # Text model
    llm_mapping = build_llm_mapping()
    process_text_gguf(args.gguf_llm, llm_mapping, writer)

    # Vision
    if args.gguf_vision:
        vision_mapping = build_vision_mapping()
        process_vision_gguf(args.gguf_vision, vision_mapping, writer)

    # MTP (downloads only the specific shards needed, not the full model)
    if not args.skip_mtp:
        copy_mtp_from_reference(args.reference_repo, writer)

    # Finalize
    print("\nFinalizing shards...")
    n_shards = writer.finalize()

    # Verify
    print("\nVerifying against reference...")
    missing, extra = verify_against_reference(ref_snapshot, set(writer.weight_map.keys()))

    print(f"\nConversion complete: {len(writer.weight_map)} tensors in {n_shards} shards, "
          f"{writer.total_size / 1e9:.2f} GB")
    if missing:
        print("WARNING: Not verification-clean -- fix mapping before quantizing.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
