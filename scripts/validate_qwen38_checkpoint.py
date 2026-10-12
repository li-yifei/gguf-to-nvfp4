#!/usr/bin/env python3
"""CPU-only header/index audit; optional same-source BF16 MTP/vision comparison."""
import argparse
import json
import struct
from pathlib import Path


def inspect(root):
    root = Path(root)
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    tensors = {}
    for name in sorted(set(weight_map.values())):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Shard outside checkpoint: {name}')
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if size > 100_000_000:
                raise ValueError(f'Invalid header size: {name}')
            header = json.loads(f.read(size))
        payload_size = path.stat().st_size - 8 - size
        for key, info in header.items():
            if key == '__metadata__':
                continue
            if key in tensors or weight_map.get(key) != name:
                raise ValueError(f'Duplicate or incorrectly indexed tensor: {key}')
            start, end = info['data_offsets']
            if not 0 <= start <= end <= payload_size:
                raise ValueError(f'Invalid tensor offsets: {key}')
            tensors[key] = (path, 8 + size + start, end - start, info)
    if set(tensors) != set(weight_map):
        raise ValueError('Index/header keys differ')
    return tensors


def same_bytes(left, right):
    lp, lo, ln, li = left
    rp, ro, rn, ri = right
    if ln != rn or li['dtype'] != ri['dtype'] or li['shape'] != ri['shape']:
        return False
    with lp.open('rb') as lf, rp.open('rb') as rf:
        lf.seek(lo)
        rf.seek(ro)
        remaining = ln
        while remaining:
            count = min(remaining, 4 * 1024 * 1024)
            if lf.read(count) != rf.read(count):
                return False
            remaining -= count
    return True


def validate(root, source=None):
    root = Path(root)
    tensors = inspect(root)
    config = json.loads((root / 'config.json').read_text())['quantization_config']
    if config.get('quant_method') != 'compressed-tensors' or config.get('format') != 'nvfp4-pack-quantized':
        raise ValueError('Expected compressed-tensors nvfp4-pack-quantized')
    mtp = [k for k in tensors if k.startswith('mtp.')]
    vision = [k for k in tensors if k.startswith(('model.visual.', 'visual.'))]
    if len(mtp) != 15 or len(vision) != 333:
        raise ValueError(f'Expected Qwen3.8-27B full VLM: MTP=15, vision=333; got {len(mtp)}, {len(vision)}')
    for key in mtp + vision:
        if tensors[key][3]['dtype'] != 'BF16':
            raise ValueError(f'Expected BF16: {key}')
    if not any(k.endswith('.weight_packed') for k in tensors):
        raise ValueError('No packed quantized weights')
    if source:
        original = inspect(source)
        for key in mtp + vision:
            if key not in original or not same_bytes(tensors[key], original[key]):
                raise ValueError(f'Source mismatch: {key}')
    return {'tensor_count': len(tensors), 'mtp_tensor_count': len(mtp),
            'vision_tensor_count': len(vision), 'format': config['format'],
            'same_source_bytes_verified': bool(source), 'runtime_tested': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--source', help='Original same-variant BF16 checkpoint')
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.checkpoint, args.source), indent=2))
    except (OSError, ValueError, KeyError, struct.error) as exc:
        parser.exit(1, f'Checkpoint validation failed: {exc}\n')
