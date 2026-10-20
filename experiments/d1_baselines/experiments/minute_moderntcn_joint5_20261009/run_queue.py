"""GPU0 concurrent training of the prespecified one-minute joint five-channel jobs."""
from if_release.runtime import loads as _release_json_loads
from pathlib import Path
from datetime import datetime, timezone, timedelta
import getpass
import json
import os
import subprocess
import sys
import time
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TZ = timezone(timedelta(hours=8))

def now():
    return datetime.now(TZ)

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def gpu_state(gpu):
    fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    uuid, free_mib = (fields[0].strip(), int(fields[1].strip()))
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
    processes = []
    for row in rows:
        values = [x.strip() for x in row.split(',')]
        if values[0] == uuid:
            owner_result = subprocess.run(['ps', '-p', values[1], '-o', 'user='], capture_output=True, text=True)
            if owner_result.returncode == 1:
                print('GPU_PID_EXITED_BEFORE_OWNER_QUERY ' + values[1], flush=True)
                continue
            owner_result.check_returncode()
            owner = owner_result.stdout.strip()
            processes.append(dict(pid=int(values[1]), owner=owner, memory_mib=int(values[2])))
    return dict(gpu=gpu, free_mib=free_mib, processes=processes)

def ordered_groups(plan, configurations):
    groups = []
    for families in plan['scheduling_groups']:
        jobs = []
        for seed in plan['seeds']:
            for family in families:
                jobs.extend((item for item in configurations if item['config']['model'] == family and item['config']['seed'] == seed))
        groups.append(jobs)
    return groups

def archive_job(item, returncode, records):
    job, cfg, directory = (item['job'], item['config'], item['directory'])
    ended = now()
    job.update(status='completed' if returncode == 0 else 'failed', returncode=returncode, ended_at=ended.isoformat(), wall_seconds=(ended - item['started']).total_seconds())
    (directory / 'exit_code.txt').write_text(str(returncode), encoding='utf-8')
    record = records / (ended.strftime('%Y-%m-%d_%H%M%S') + '_' + cfg['run_name'] + '_' + job['status'] + '.md')
    text = '# 一分钟温度与四位移联合预测单卡并发实验：' + cfg['model'] + f"\n\n- 项目：IFpaper_v1；开始 {job['started_at']}，结束 {job['ended_at']}；seed {cfg['seed']}；GPU {job['gpu']}，训练 PID {job['pid']}，返回码 {returncode}。\n- 目的：ModernTCN五种子全新训练；冻结150训练/26验证/39测试起点和实测mask，预测未来24小时温度及四个位移。\n- 操作：1分钟采样，历史30天43200×5输入，未来24小时1440×5输出；仅历史温度＋四位移，未来实测值和日历不进入模型。统一标准化mask MSE，两项差分权重均为0，原始预测不后处理滤波。验证集联合标准化RMSE选模；测试集不用于调配置或安排训练顺序。\n- 入口：{HERE / 'train.py'}；配置：{HERE / item['filename']}；数据说明：{HERE / 'dataset/README.md'}。\n- 单卡并发预算：本组 {cfg['gpu_budget_mib']} MiB；这是准入预算，不是实测峰值；并发任务共享算力，耗时不等于独占卡速度。\n- 输出：{directory}；日志 train.log，状态 job_status.json；summary.json、suite_results.json、预测 NPZ 为结果依据。\n"
    if returncode == 0:
        summary = _release_json_loads((directory / 'summary.json').read_text())
        accuracy = _release_json_loads((directory / 'suite_results.json').read_text())['results']['accuracy']
        text += f"- 结果：参数 {summary['parameters']}，最佳验证轮次 {summary['best_accuracy_epoch']}，共 {summary['epochs_ran']} 轮；验证四位移 pooled RMSE {accuracy['validation']['pooled_rmse_mm']:.6f} mm，温度 RMSE {accuracy['validation']['temperature_rmse_c']:.6f} ℃，联合标准化RMSE {accuracy['validation']['joint_standardized_rmse']:.6f}；测试四位移 pooled RMSE {accuracy['test']['pooled_rmse_mm']:.6f} mm，温度 RMSE {accuracy['test']['temperature_rmse_c']:.6f} ℃；训练循环计时 {summary['training_seconds']:.1f} 秒，训练及导出计时 {summary['elapsed_seconds']:.1f} 秒，完整进程计时 {job['wall_seconds']:.1f} 秒；本进程峰值 CUDA allocated {summary['peak_cuda_allocated_mib']:.1f} MiB。\n"
    else:
        text += '- 结果：训练失败，原始错误保留在 train.log；没有自动重试或更换配置。\n'
    record.write_text(text, encoding='utf-8')
    job['record'] = str(record)
    dump(directory / 'job_status.json', job)

def main():
    plan = _release_json_loads((HERE / 'plan.json').read_text())
    out = PROJECT / plan['output']
    records = PROJECT / 'outputs/experiment_records'
    out.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    configurations = [dict(filename=filename, config=_release_json_loads((HERE / filename).read_text())) for filename in plan['configs']]
    groups = ordered_groups(plan, configurations)
    gpu = plan['gpu']
    state = dict(controller_pid=os.getpid(), controller_owner=getpass.getuser(), started_at=now().isoformat(), gpu=gpu, max_workers=plan['max_workers'], memory_budget_mib=plan['memory_budget_mib'], seeds=plan['seeds'], status='running', planned_jobs=len(configurations), jobs=[], reused_success=0, final_expected_jobs=len(configurations), reuse_runs=[], scheduling_groups=plan['scheduling_groups'])
    dump(out / 'queue_status.json', state)
    active = []
    for group_index, pending in enumerate(groups):
        state['phase'] = group_index + 1
        state['phase_models'] = plan['scheduling_groups'][group_index]
        state['resource_wait'] = None
        dump(out / 'queue_status.json', state)
        while pending or active:
            for item in list(active):
                returncode = item['process'].poll()
                if returncode is not None:
                    item['logfile'].close()
                    archive_job(item, returncode, records)
                    active.remove(item)
                    dump(out / 'queue_status.json', state)
                    print('ENDED ' + json.dumps(item['job']), flush=True)
            resource = gpu_state(gpu)
            own_pids = {item['process'].pid for item in active}
            external = [process for process in resource['processes'] if process['pid'] not in own_pids]
            state['resource_snapshot'] = resource
            reserved = sum((item['config']['gpu_budget_mib'] for item in active))
            if external and pending:
                wait = dict(reason='external_gpu_process', processes=external)
            elif pending and len(active) < plan['max_workers']:
                next_budget = pending[0]['config']['gpu_budget_mib']
                wait = dict(reason='memory_budget', reserved_mib=reserved, requested_mib=next_budget, free_mib=resource['free_mib']) if reserved + next_budget > plan['memory_budget_mib'] or next_budget > resource['free_mib'] else None
            else:
                wait = None
            if wait != state['resource_wait']:
                state['resource_wait'] = wait
                print('RESOURCE_STATE ' + json.dumps(wait), flush=True)
            if pending and len(active) < plan['max_workers'] and (not external) and (wait is None):
                item = pending.pop(0)
                cfg = item['config']
                directory = PROJECT / cfg['output']
                directory.mkdir(parents=True, exist_ok=True)
                started = now()
                logfile = (directory / 'train.log').open('w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS=str(cfg['torch_threads']), MKL_NUM_THREADS=str(cfg['torch_threads']))
                process = subprocess.Popen([sys.executable, '-u', str(HERE / 'train.py'), '--config', str(HERE / item['filename'])], cwd=PROJECT, env=env, stdout=logfile, stderr=subprocess.STDOUT)
                job = dict(name=cfg['run_name'], model=cfg['model'], seed=cfg['seed'], gpu=gpu, pid=process.pid, started_at=started.isoformat(), status='running', phase=group_index + 1, gpu_budget_mib=cfg['gpu_budget_mib'], output=str(directory), log=str(directory / 'train.log'))
                item.update(process=process, logfile=logfile, directory=directory, started=started, job=job)
                state['jobs'].append(job)
                active.append(item)
                dump(directory / 'job_status.json', job)
                print('STARTED ' + json.dumps(job), flush=True)
            state['counts'] = dict(success=sum((j['status'] == 'completed' for j in state['jobs'])), failed=sum((j['status'] == 'failed' for j in state['jobs'])), running=len(active), not_started=len(configurations) - len(state['jobs']))
            dump(out / 'queue_status.json', state)
            time.sleep(3)
    completed = [job['name'] for job in state['jobs'] if job['status'] == 'completed']
    state.update(status='summarizing', ended_at=now().isoformat())
    dump(out / 'queue_status.json', state)
    result = subprocess.run([sys.executable, str(HERE / 'compare.py'), *completed], cwd=PROJECT)
    state.update(status=('completed' if len(completed) == len(configurations) else 'finished_with_failures') if result.returncode == 0 else 'summary_failed', summary_returncode=result.returncode, ended_at=now().isoformat())
    dump(out / 'queue_status.json', state)
    print('QUEUE_FINISHED ' + json.dumps(state), flush=True)
if __name__ == '__main__':
    main()
