import csv
import json
import random
import numpy as np
import torch

def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def write_history(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
