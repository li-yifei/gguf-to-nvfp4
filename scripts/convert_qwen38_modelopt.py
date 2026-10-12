#!/usr/bin/env python3
"""Convert ModelOpt NVFP4 to compressed-tensors; retain unquantized state."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve() or (dst.exists() and (not dst.is_dir() or any(dst.iterdir()))):
        parser.error('destination must be empty and different from source')
    if args.workers < 1:
        parser.error('--workers must be positive')
    index = json.loads((src / 'model.safetensors.index.json').read_text())
    keys = set(index['weight_map'])
    # ModelOpt-quantized modules are exactly those carrying input_scale.
    quantized = {k.removesuffix('.input_scale') for k in keys if k.endswith('.input_scale')}
    if not quantized:
        parser.error('No ModelOpt input_scale tensors found')
    # Exact names keep quantized child Linear modules eligible.
    all_modules = {k.rsplit('.', 1)[0] for k in keys if '.' in k}
    ignore_modules = sorted(all_modules - quantized)
    ignore = [
        'lm_head', 'model.language_model.embed_tokens',
        're:.*visual.*', 're:.*conv1d.*', 're:^mtp.*', *ignore_modules,
    ]
    from compressed_tensors.entrypoints.convert import convert_checkpoint
    from compressed_tensors.entrypoints.convert.converters.modelopt_nvfp4 import ModelOptNvfp4Converter

    print(f'quantized_modules={len(quantized)} ignore_modules={len(ignore_modules)}')
    converter = ModelOptNvfp4Converter(targets=['Linear'], ignore=ignore)
    convert_checkpoint(args.src, args.dst, converter, max_workers=args.workers)
    # Preserve assets without overwriting converted quantization metadata.
    excluded = {'config.json', 'model.safetensors.index.json', 'hf_quant_config.json'}
    for item in src.iterdir():
        if (item.is_file() and item.suffix in {'.json', '.jinja', '.txt', '.model'}
                and item.name not in excluded and not (dst / item.name).exists()):
            shutil.copy2(item, dst / item.name)
    print(dst)


if __name__ == '__main__':
    main()
