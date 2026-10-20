from pathlib import Path
import json, zipfile
ROOT = Path(__file__).resolve().parents[1]
files = ['README.md', 'requirements.txt', 'generation_config.json', 'sensors.json', 'thermal_response.json', 'scripts/generate_dataset.py', 'data/README.md', 'data/minute_synthetic.npz', 'data/summary.json', 'data/window_origins.json', 'data/extreme_validation.json', 'figures/dataset_preview_300dpi.png', 'figures/dataset_preview.pdf', 'figures/response_structure.json']
path = ROOT / 'synthetic_deflection_public_v1.zip'
with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for relative in files:
        archive.write(ROOT / relative, 'synthetic_deflection_v1/' + relative)
(ROOT / 'public_package_contents.json').write_text(json.dumps(dict(file=path.name, bytes=path.stat().st_size, members=files), indent=2), encoding='utf-8')
print(json.dumps(dict(zip=str(path), bytes=path.stat().st_size, members=len(files))))
