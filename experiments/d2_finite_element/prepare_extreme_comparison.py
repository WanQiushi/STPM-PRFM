from if_release.runtime import loads as _release_json_loads
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
z = np.load(ROOT / 'data/minute_synthetic.npz')
q = z['thermal_states_c'].astype(float)
sensors = _release_json_loads((ROOT / 'sensors.json').read_text())
coeff = _release_json_loads((ROOT / 'thermal_node_coefficients.json').read_text())
clean = z['clean_values']
selected = {'THERM_DATA_COLD': int(np.argmax(clean[:, 4])), 'THERM_DATA_WARM': int(np.argmin(clean[:, 4]))}
text = (ROOT / 'fe_run_v2/THERMAL_BASIS.inp').read_text(encoding='ascii')
checks = {}
for name, k in selected.items():
    g, c = q[k] - 29
    text += '*Step, name=' + name + ', nlgeom=YES, inc=200\n*Static\n0.2, 1., 1.e-6, 1.\n*Temperature, op=NEW\n'
    text += ''.join(('BRIDGE_MODEL-1.%s, %.12g\n' % (n, g * a + c * b) for n, (a, b) in coeff.items()))
    text += '*Output, field, frequency=1\n*Node Output, nset=N_SIM_DEFLECTION\nU\n*Output, history, frequency=0\n*End Step\n'
    checks[name] = dict(minute_index=k, girder_delta_c=float(g), cable_delta_c=float(c), linear_clean_mm=clean[k, 1:].astype(float).tolist())
run = ROOT / 'fe_data_extremes'
run.mkdir()
(run / 'THERMAL_EXTREMES.inp').write_text(text, encoding='ascii')
(ROOT / 'data_extreme_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
for name in ['thermal_driver.py', 'run_thermal.py', 'run_thermal.cmd']:
    content = (ROOT / 'fe_run_v2' / name).read_text(encoding='ascii')
    content = content.replace('fe_run_v2\\', 'fe_data_extremes\\').replace('fe_run_v2/', 'fe_data_extremes/')
    if name == 'thermal_driver.py':
        content = content.replace("options['user']=r'" + str(run / 'source_gradient.for').replace('\\', '/') + "'", "options['user']=r'" + str(ROOT / 'fe_run_v2/source_gradient.for').replace('\\', '/') + "'")
        content = content.replace('THERMAL_BASIS', 'THERMAL_EXTREMES')
    (run / name).write_text(content, encoding='ascii')
print(json.dumps(checks))
