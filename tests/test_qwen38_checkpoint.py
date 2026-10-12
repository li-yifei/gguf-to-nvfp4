"""Small synthetic checkpoints exercise corruption and same-source preservation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from validate_qwen38_checkpoint import validate


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / 'source', self.root / 'target'
        self.tensors = {f'mtp.layer{i}.weight': torch.tensor([i], dtype=torch.bfloat16) for i in range(15)}
        self.tensors.update({f'model.visual.layer{i}.weight': torch.tensor([i], dtype=torch.bfloat16) for i in range(333)})
        self.tensors['model.language_model.layer.weight_packed'] = torch.tensor([1], dtype=torch.uint8)
        self.write(self.source, self.tensors)
        self.write(self.target, self.tensors)

    def write(self, path, tensors):
        path.mkdir(exist_ok=True)
        save_file(tensors, path / 'weights.safetensors')
        (path / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': {k: 'weights.safetensors' for k in tensors}}))
        (path / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'compressed-tensors', 'format': 'nvfp4-pack-quantized'}}))

    def restore(self):
        return subprocess.run([sys.executable, str(SCRIPTS / 'restore_qwen38_mtp.py'), str(self.source), str(self.target)], capture_output=True, text=True)

    def test_valid_and_same_source(self):
        self.assertTrue(validate(self.target, self.source)['same_source_bytes_verified'])

    def test_missing_shard(self):
        (self.target / 'weights.safetensors').unlink()
        with self.assertRaises(FileNotFoundError):
            validate(self.target)

    def test_wrong_index(self):
        p = self.target / 'model.safetensors.index.json'
        index = json.loads(p.read_text())
        del index['weight_map']['mtp.layer0.weight']
        p.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'incorrectly indexed'):
            validate(self.target)

    def test_wrong_variant_rejected(self):
        changed = dict(self.tensors)
        changed['mtp.layer0.weight'] = torch.tensor([-1], dtype=torch.bfloat16)
        self.write(self.target, changed)
        with self.assertRaisesRegex(ValueError, 'Source mismatch'):
            validate(self.target, self.source)
        result = self.restore()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('differs from source', result.stderr)

    def test_partial_mtp_restored_and_idempotent(self):
        self.write(self.target, {k: v for k, v in self.tensors.items() if k != 'mtp.layer0.weight'})
        result = self.restore()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(validate(self.target, self.source)['same_source_bytes_verified'])
        before = (self.target / 'model.safetensors.index.json').read_bytes()
        result = self.restore()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, (self.target / 'model.safetensors.index.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
