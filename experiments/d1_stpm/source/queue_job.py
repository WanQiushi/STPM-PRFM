def run_job(name, command, env, stage, gpu):
    with (LOGS / (name + '.log')).open('w') as log:
        with LOCK:
            if STOP.is_set():
                return None
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (LOGS / (name + '.pid')).write_text(str(process.pid))
            ACTIVE[str(gpu)] = {'run': name, 'stage': stage, 'child_pid': process.pid, 'physical_gpu': gpu, 'log': str(LOGS / (name + '.log'))}
            progress()
        print('START ' + name + ' GPU ' + str(gpu) + ' PID ' + str(process.pid), flush=True)
        code = process.wait()
    (LOGS / (name + '.exit')).write_text(str(code))
    with LOCK:
        del ACTIVE[str(gpu)]
        if code != 0:
            FAILED.append({'run': name, 'stage': stage, 'exit': code, 'physical_gpu': gpu})
            STOP.set()
            (OUT / (name + '_failure_record.md')).write_text('# va core ablation failed\n\n- Date: ' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat() + '; IFpaper_v1.\n- Purpose:remaining A1/A2/A4/A5/A6/A7 core ablations, dual GPU queue, five temperature seeds.\n- Stage/job: ' + stage + '/' + name + '; GPU ' + str(gpu) + '; PID ' + str(process.pid) + '; exit ' + str(code) + '.\n- Exact command: ' + json.dumps(command) + '.\n- Log: ' + str(LOGS / (name + '.log')) + '.\n- Products: ' + str(OUT) + '; stop assigning new stages, retain finished outputs and let already-running child finish; no retry.\n', encoding='utf-8')
        else:
            FINISHED.append(name)
        progress()
    print(('FAILED ' if code else 'DONE ') + name + ' GPU ' + str(gpu), flush=True)
    return code
