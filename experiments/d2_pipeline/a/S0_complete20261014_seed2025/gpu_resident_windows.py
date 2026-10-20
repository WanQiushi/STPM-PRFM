"""Materialize the existing dataset fields on CUDA, preserving order and dtype."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class ResidentWindows(Dataset):

    def __init__(self, dataset, device):
        rows = [dataset[index] for index in range(len(dataset))]
        self.arrays = {key: torch.as_tensor(np.stack([row[key] for row in rows]), device=device) for key in rows[0]}
        self.resident_bytes = sum((value.numel() * value.element_size() for value in self.arrays.values()))
        print('RESIDENT_DATA ' + json.dumps({'source': type(dataset).__name__, 'windows': len(dataset), 'bytes': self.resident_bytes, 'fields': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)} for key, value in self.arrays.items()}}), flush=True)

    def __len__(self):
        return len(self.arrays['origin'])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}
