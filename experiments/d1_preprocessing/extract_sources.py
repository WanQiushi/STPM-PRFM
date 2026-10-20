"""Extract the seven selected sensors, retaining the source CSV text fields."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
destination = root / 'data' / 'sources'
destination.mkdir(parents=True, exist_ok=True)
selection = {'温度.csv': ['Temperature'], '支座位移.csv': ['Pier1', 'Pier2', 'Pier3', 'Pier4'], '梁端位移.csv': ['Auxiliary1', 'Auxiliary2']}
manifest = []
for filename, ids in selection.items():
    source = args.source_root / filename
    output = destination / (filename + '.gz')
    source_rows = selected_rows = 0
    with gzip.open(output, 'wt', encoding='utf-8', newline='') as stream:
        stream.write('MeasureNum,value,MonitorTime\n')
        for chunk in pd.read_csv(source, dtype=str, keep_default_na=False, chunksize=400000):
            source_rows += len(chunk)
            selected = chunk.loc[chunk.MeasureNum.isin(ids), ['MeasureNum', 'value', 'MonitorTime']]
            selected_rows += len(selected)
            selected.to_csv(stream, index=False, header=False)
    manifest.append({'original_path': str(source), 'original_bytes': source.stat().st_size, 'original_rows': source_rows, 'selected_rows': selected_rows, 'selected_ids': ids, 'archive_path': str(output.relative_to(root)).replace('\\', '/')})
    print(filename, selected_rows, flush=True)
(destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
