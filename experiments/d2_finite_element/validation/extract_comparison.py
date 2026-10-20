from __future__ import print_function
from if_release.runtime import expand as _release_expand, load as _release_json_load
import json, os
from odbAccess import openOdb
ROOT = _release_expand('${FE_WORK}/d2')
sensors = _release_json_load(open(os.path.join(ROOT, 'dataset', 'sensors.json')))
checks = _release_json_load(open(os.path.join(ROOT, 'dataset', 'data_extreme_checks.json')))
odb = openOdb(os.path.join(ROOT, 'fe_extreme_validation', 'THERM_D21_EXTREMES.odb'), readOnly=True)

def displacement(step):
    field = odb.steps[step].frames[-1].fieldOutputs['U']
    u = {v.nodeLabel: float(v.data[2]) * 1000.0 for v in field.values if v.instance.name == 'BRIDGE_MODEL-1'}
    return [sum((u[n] * w for n, w in zip(s['nodes'], s['weights']))) for s in sensors]
base = displacement('EQCHECK_STRICT')
for name, check in checks.items():
    actual = [a - b for a, b in zip(displacement(name), base)]
    error = [a - b for a, b in zip(actual, check['linear_clean_mm'])]
    check['nonlinear_mm'] = actual
    check['error_mm'] = error
    check['relative_l2'] = (sum((e * e for e in error)) / sum((a * a for a in actual))) ** 0.5
odb.close()
with open(os.path.join(ROOT, 'dataset', 'data', 'extreme_validation.json'), 'w') as out:
    json.dump(checks, out, indent=2)
print(json.dumps({k: dict(relative_l2=v['relative_l2'], max_absolute_error_mm=max((abs(e) for e in v['error_mm']))) for k, v in checks.items()}))
