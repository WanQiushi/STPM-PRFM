from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
import json, shutil, bisect, re
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(_release_expand('${FE_MODEL}'))
shutil.copytree(SOURCE, ROOT / 'full_model')
run = ROOT / 'fe_run'
run.mkdir()
shutil.copy2(SOURCE / 'source_gradient_table.txt', run / 'source_gradient_table.txt')
fortran = (SOURCE / 'source_gradient.for').read_text(encoding='utf-8')
fortran = re.sub("FILE='[^']+'", "FILE='source_gradient_table.txt'", fortran)
(run / 'source_gradient.for').write_text(fortran, encoding='ascii')
d = _release_json_loads((SOURCE / 'sources/BRIDGE_MODEL_source.json').read_text(encoding='utf-8'))['DBG']
groups = {v['NAME']: v for v in d['GRUP'].values()}
line = sorted([(float(d['NODE'][str(n)]['X']), n) for n in groups['MAIN_SPAN_GIRDER']['N_LIST'] if abs(d['NODE'][str(n)]['Y'] + 21.5) < 0.01])
xs = [p[0] for p in line]
sensors = []
for j in range(1, 8):
    x = 168 + 1092 * j / 8
    index = bisect.bisect_left(xs, x)
    if xs[index] == x:
        ns = [line[index][1]]
        weights = [1.0]
    else:
        left, right = (line[index - 1], line[index])
        r = (x - left[0]) / (right[0] - left[0])
        ns = [left[1], right[1]]
        weights = [1 - r, r]
    sensors.append(dict(channel='SIM_D%02d' % j, span_fraction=j / 8, x_m=x, y_m=-21.5, nodes=ns, weights=weights))
labels = sorted(set((n for sensor in sensors for n in sensor['nodes'])))
coeff = {int(n): [0.0, 0.0] for n in d['NODE']}
for n in groups['STIFFENING_GIRDER']['N_LIST']:
    coeff[n] = [1.0, 0.0]
for n in groups['MAIN_CABLE']['N_LIST']:
    coeff[n] = [0.0, 1.0]
hanger_lines = {}
for n in groups['HANGER']['N_LIST']:
    coord = d['NODE'][str(n)]
    key = (round(coord['X'], 5), -1 if coord['Y'] < 0 else 1)
    hanger_lines.setdefault(key, []).append(n)
for nodes in hanger_lines.values():
    zs = [d['NODE'][str(n)]['Z'] for n in nodes]
    low, high = (min(zs), max(zs))
    for n in nodes:
        r = (d['NODE'][str(n)]['Z'] - low) / (high - low)
        coeff[n] = [1 - r, r]
(ROOT / 'sensors.json').write_text(json.dumps(sensors, indent=2), encoding='utf-8')
(ROOT / 'thermal_node_coefficients.json').write_text(json.dumps(coeff, separators=(',', ':')), encoding='utf-8')
config = dict(version='synthetic_deflection_abaqus_v1_20261009', source_model=str(SOURCE), start_beijing='2026-10-09T21:29:24', model_scope='Linearized thermal response about the frozen bridge equilibrium; not a thermomechanical constitutive hysteresis model', reference_temperature_c=29.0, numeric_temperature_reference_c=0.0, girder_tau_hours=2.0, cable_tau_hours=4.0, tau_source='User-approved public simulation assumptions, not measured bridge parameters', dummy_material_id=6, dummy_original_expansion_per_c=1.2e-05, dummy_derived_expansion_per_c=0.0, dummy_reason='Rigid dummy connectors represent kinematic connections, not independently expanding physical members', cpu_count=1, memory_limit_mb=8000, fe_steps=['EQCHECK_STRICT', 'THERM_G_UNIT', 'THERM_C_UNIT', 'THERM_BOTH_PLUS10', 'THERM_BOTH_MINUS10'], deflection_positive_direction='global +Z, upward', synthetic_side='model Y=-21.5 m; not an independently verified upstream sensor mapping', sensor_interpolation='Piecewise linear interpolation between adjacent original upper-chord nodes at the seven main-span fractions', days=400, split_days=[280, 60, 60], minute_step=1, seed=20241009, warmup_days=14, seasonal_amplitude_c=6.0, daily_mean_amplitude_c=3.0, weather_ar_timescale_hours=72.0, weather_stationary_std_c=1.2, temperature_noise_std_c=0.05, deflection_noise_std_mm=0.5, noise_source='Independent Gaussian observation noise, public simulation assumptions', future_mechanical_memory='Two first-order thermal states; no added BW state and no neural generator', thermal_field='Girder and cable uniform group temperatures; hanger nodal temperatures linearly interpolate by height; towers/piers/other nodes remain at zero temperature increment')
(ROOT / 'generation_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
base = (SOURCE / 'BRIDGE_MODEL.inp').read_text(encoding='utf-8')
base = base[:base.index('** Step87 repeats')]
base = base.replace('*Material, name=MAT_M06\n*Elastic\n21000000000000, 0.3\n*Expansion\n1.2e-05,', '*Material, name=MAT_M06\n*Elastic\n21000000000000, 0.3\n*Expansion\n0.,')
setblock = '*Nset, nset=N_SIM_DEFLECTION, instance=BRIDGE_MODEL-1\n' + ', '.join(map(str, labels)) + '\n'
base = base.replace('*Initial Conditions, type=STRESS, user, unbalanced stress=STEP', setblock + '*Initial Conditions, type=STRESS, user, unbalanced stress=STEP')
text = base
for name, g, c, perturb in [('THERM_G_UNIT', 1.0, 0.0, True), ('THERM_C_UNIT', 0.0, 1.0, True), ('THERM_BOTH_PLUS10', 10.0, 10.0, False), ('THERM_BOTH_MINUS10', -10.0, -10.0, False)]:
    if perturb:
        text += '*Step, name=' + name + ', perturbation\n*Static\n'
    else:
        text += '*Step, name=' + name + ', nlgeom=YES, inc=200\n*Static\n0.2, 1., 1.e-6, 1.\n'
    text += '*Temperature, op=NEW\n'
    text += ''.join(('BRIDGE_MODEL-1.%d, %.12g\n' % (n, g * a + c * b) for n, (a, b) in sorted(coeff.items())))
    text += '*Output, field, frequency=1\n*Node Output, nset=N_SIM_DEFLECTION\nU\n*Output, history, frequency=0\n*End Step\n'
(run / 'THERMAL_BASIS.inp').write_text(text, encoding='ascii')
driver = (SOURCE / 'driver_template.py').read_text(encoding='utf-8')
pos = driver.index("options['user']")
driver = driver[:pos] + "options['input']=r'" + str(run / 'THERMAL_BASIS').replace('\\', '/') + "'\noptions['job']='THERMAL_BASIS'\noptions['user']=r'" + str(run / 'source_gradient.for').replace('\\', '/') + "'\noptions['cpus']=1\noptions['memory']='8000mb'\nanalysis=StandardAnalysis(options)\nstatus=analysis.run()\nsys.exit(status)\n"
(run / 'thermal_driver.py').write_text(driver, encoding='ascii')
runpy = _release_expand("import os\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nos.chdir(str(root))\nos.environ['PATH']=r'${COMPILER}/compiler/2024.0/bin;${MSVC}/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64;'+os.environ['PATH']\ncode=(root/'thermal_driver.py').read_text(encoding='utf-8')\nexec(compile(code,str(root/'thermal_driver.py'),'exec'))\n")
(run / 'run_thermal.py').write_text(runpy, encoding='ascii')
cmd = _release_expand('@echo off\ncall "${MSVC}\\VC\\Auxiliary\\Build\\vcvars64.bat" > nul\ncall "${COMPILER}\\compiler\\2024.0\\env\\vars.bat" intel64 vs2019 > nul\ncall "${ABAQUS}\\abaqus.bat" python "') + str(run / 'run_thermal.py') + '"\n'
(run / 'run_thermal.cmd').write_text(cmd, encoding='ascii')
summary = dict(source_nodes=len(d['NODE']), source_elements=len(d['ELEM']), main_span_m=1092, deflection_sensors=sensors, model_copy=str(ROOT / 'full_model'), thermal_input=str(run / 'THERMAL_BASIS.inp'), thermal_input_bytes=(run / 'THERMAL_BASIS.inp').stat().st_size, original_source_unchanged=True)
(ROOT / 'extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(dict(root=str(ROOT), nodes=summary['source_nodes'], elements=summary['source_elements'], sensor_nodes=labels, thermal_input_bytes=summary['thermal_input_bytes']), ensure_ascii=False))
