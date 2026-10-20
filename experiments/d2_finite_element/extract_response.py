from __future__ import print_function
from if_release.runtime import load as _release_json_load
import os, json
from odbAccess import openOdb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sensors = _release_json_load(open(os.path.join(ROOT, 'sensors.json')))
odb = openOdb(os.path.join(ROOT, 'fe_run_v2', 'THERMAL_BASIS.odb'), readOnly=True)
results = {}
for name in ['EQCHECK_STRICT', 'THERM_G_UNIT', 'THERM_C_UNIT', 'THERM_BOTH_PLUS10', 'THERM_BOTH_MINUS10']:
    f = odb.steps[name].frames[-1].fieldOutputs['U']
    nodes = {v.nodeLabel: float(v.data[2]) * 1000.0 for v in f.values if v.instance.name == 'BRIDGE_MODEL-1'}
    results[name] = [sum((nodes[n] * w for n, w in zip(s['nodes'], s['weights']))) for s in sensors]
base = results['EQCHECK_STRICT']
for name in ['THERM_BOTH_PLUS10', 'THERM_BOTH_MINUS10']:
    results[name] = [a - b for a, b in zip(results[name], base)]
g, c = (results['THERM_G_UNIT'], results['THERM_C_UNIT'])
comparisons = {}
for name, amp in [('THERM_BOTH_PLUS10', 10.0), ('THERM_BOTH_MINUS10', -10.0)]:
    predicted = [amp * (a + b) for a, b in zip(g, c)]
    actual = results[name]
    error = [a - b for a, b in zip(actual, predicted)]
    comparisons[name] = dict(nonlinear_mm=actual, linear_mm=predicted, error_mm=error, relative_l2=(sum((e * e for e in error)) / sum((a * a for a in actual))) ** 0.5)
out = dict(channels=[s['channel'] for s in sensors], units='mm per degC', girder_mm_per_c=g, cable_mm_per_c=c, base_equilibrium_increment_mm=base, reference_temperature_c=29.0, positive_direction='global +Z (upward)', linearization_checks=comparisons, description='Perturbation U fields are increments about the base state; nonlinear results subtract the equilibrium U. This is a linearized thermal response map, not a general nonlinear surrogate.')
with open(os.path.join(ROOT, 'thermal_response.json'), 'w') as stream:
    json.dump(out, stream, indent=2)
odb.close()
print(json.dumps(out))
