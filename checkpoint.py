"""Load trusted full training checkpoints across old and new PyTorch versions."""
import inspect
import torch

def load_checkpoint(path, device):
    kwargs = {'map_location': device}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)
