"""Accept known checkpoint containers/prefixes; never discard missing or bad tensors."""
import hashlib
import json
from pathlib import Path

CLASSES = ['Common_Rust', 'Gray_Leaf_Spot', 'Healthy', 'Northern_Leaf_Blight']

class CheckpointError(ValueError):
    pass


def normalize_state(checkpoint, expected_classes=CLASSES):
    import torch
    if not isinstance(checkpoint, dict):
        raise CheckpointError('Checkpoint must be a state dictionary')
    for field in ('classes', 'class_names'):
        if field in checkpoint and checkpoint[field] != expected_classes:
            raise CheckpointError('Checkpoint class order differs from the application')
    state = checkpoint
    for field in ('state_dict', 'model_state_dict'):
        if field in checkpoint:
            state = checkpoint[field]
            break
    if not isinstance(state, dict) or not state:
        raise CheckpointError('Empty or invalid state dictionary')
    while True:
        prefix = next((p for p in ('module.', '_orig_mod.')
                       if all(isinstance(k, str) and k.startswith(p) for k in state)), None)
        if prefix is None:
            break
        state = {k[len(prefix):]: v for k, v in state.items()}
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise CheckpointError('State dictionary contains non-tensor weights')
    return state


def validate_state(model, state):
    import torch
    expected = model.state_dict()
    if set(state) != set(expected):
        raise CheckpointError('Missing or unexpected checkpoint tensors')
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise CheckpointError(f'Checkpoint shape mismatch: {key}')
        if not torch.isfinite(value).all():
            raise CheckpointError(f'Non-finite checkpoint tensor: {key}')
    model.load_state_dict(state, strict=True)


def load_checked(model, path, classes):
    import torch
    if classes != CLASSES:
        raise CheckpointError('Report class order differs from the application')
    manifest = json.loads((Path(path).parent / 'manifest.json').read_text())
    entry = next((f for f in manifest['files'] if f['filename'] == Path(path).name), None)
    if entry is None:
        raise CheckpointError('Checkpoint is not registered in the manifest')
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            digest.update(chunk)
    if Path(path).stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
        raise CheckpointError('Checkpoint checksum or size does not match the manifest')
    validate_state(model, normalize_state(torch.load(path, map_location='cpu', weights_only=True), classes))
    return digest.hexdigest()
