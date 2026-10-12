#!/usr/bin/env python3
"""Restore missing MTP from the exact BF16 source used for quantization."""
import argparse
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from validate_qwen38_checkpoint import inspect, same_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('checkpoint')
    args = p.parse_args()
    source, target = Path(args.source), Path(args.checkpoint)
    original, current = inspect(source), inspect(target)
    keys = sorted(k for k in original if k.startswith('mtp.'))
    if len(keys) != 15 or any(original[k][3]['dtype'] != 'BF16' for k in keys):
        p.error('Source must contain all 15 BF16 MTP tensors')
    if set(k for k in current if k.startswith('mtp.')) - set(keys):
        p.error('Unexpected target MTP keys')
    missing = []
    for key in keys:
        if key in current:
            if not same_bytes(original[key], current[key]):
                p.error(f'Existing MTP differs from source: {key}')
        else:
            missing.append(key)
    if not missing:
        print('All 15 MTP tensors already match the source')
        return
    name = 'model-mtp-restored.safetensors'
    if (target / name).exists():
        p.error(f'Refusing to overwrite {name}')
    tensors = {}
    for key in missing:
        with safe_open(original[key][0], framework='pt', device='cpu') as f:
            tensors[key] = f.get_tensor(key)
    index_path = target / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    save_file(tensors, target / name)
    index['weight_map'].update({k: name for k in missing})
    metadata = index.setdefault('metadata', {})
    if 'total_size' in metadata:
        metadata['total_size'] += sum(original[k][2] for k in missing)
    if 'total_parameters' in metadata:
        metadata['total_parameters'] += sum(t.numel() for t in tensors.values())
    temp = index_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(index, indent=2) + '\n')
    temp.replace(index_path)
    print(f'Restored {len(missing)} same-source MTP tensors')


if __name__ == '__main__':
    main()
