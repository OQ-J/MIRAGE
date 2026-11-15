"""Inference-only reader for compatible tensor-and-metadata checkpoints."""
from pathlib import Path
import json
import re
import torch

KIND = 'v22-stride16-self-cross-register-gated-softmax-direct-velocity-rf'

def _parameter_name(name):
    """Normalize imported tensor names to the public flat block layout."""
    if not isinstance(name, str):
        return name
    return re.sub(r'^groups\.(\d+)\.0\.', r'blocks.\1.', name)


def _named_state(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key = _parameter_name(key)
            if key in out:
                raise ValueError('Conflicting imported parameter names')
            out[key] = _named_state(item)
        return out
    if isinstance(value, list):
        return [_named_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_named_state(item) for item in value)
    return _parameter_name(value)  # Tensor storage is not copied.


def normalize(saved):
    """Adapt supported checkpoint metadata and names without changing tensor values."""
    config = dict(saved['model_config'])
    stages = config.pop('groups', None)
    if stages is not None:
        if not isinstance(stages, (list, tuple)) or not stages or any(stage != 'G' for stage in stages):
            raise ValueError('Unsupported imported block layout')
        if 'num_blocks' in config and config['num_blocks'] != len(stages):
            raise ValueError('Conflicting imported layer count')
        config['num_blocks'] = len(stages)
    result = _named_state(saved)
    result['model_config'] = config
    return result


def read(path):
    path = Path(path)
    if path.is_dir():
        path = path / json.loads((path / 'latest.json').read_text())['file']
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    if not isinstance(saved, dict) or saved.get('kind') != KIND:
        raise ValueError('Expected a compatible /16 MIRAGE checkpoint')
    required = ('model', 'model_config', 'global_step')
    if any(key not in saved for key in required):
        raise ValueError('Checkpoint must contain model, model_config and global_step')
    return normalize(saved)
