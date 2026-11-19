"""Full 512-step coverage, time/environment and replay stratification."""
import runtime
import json
import os
import shutil
import subprocess
import time
from compare_parity import compare_pair


def main():
    base=runtime.ART/'coverage512-agents128-u1024';base.mkdir(parents=True,exist_ok=True)
    desktop=runtime.DESKTOP/'agents128-horizon512-stratified-u1024';desktop.mkdir(parents=True,exist_ok=True)
    map_dir=base/'maps/warehouse32'
    shutil.copytree(runtime.ART/'controlled-parity-agents128-u1024/maps/warehouse32',map_dir)
    map_metadata=json.loads((map_dir/'configuration.json').read_text())
    os.environ['JOINTLOOM_DAGGER_MAP_DIR']=str(map_dir)
    shutil.copy2(runtime.WORK/'FULL_COVERAGE_PLAN.md',desktop/'EXPERIMENT_PLAN.md')
    frozen=base/'execution-source'
    shutil.copytree(runtime.WORK,frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    protocol=dict(agents=128,envs=16,seeds=[0,1,2],rounds=32,steps=512,queries=128,
        query_sampling='stratified',query_time_bins=8,replay_sampling='episode-stratified',
        updates_per_round=32,background_graphs=512,background_agents=64,background_per_batch=48,online_per_batch=16,
        map_directory=str(map_dir),map_configuration=map_metadata,
        learning_rate=1e-6,freeze_batchnorm=True,expert_iteration_budget=2000,
        expert_processes=4,pibt_threads=10,cpu_workers=8,cpu_affinity=list(range(16)),
        deterministic=True,independent_rngs=True,proof_rounds=4,proof_updates=128,
        quality_test_seed_base=2037000000,quality_test_tasks=128,
        selection='fixed final after 1024 updates; no test-based selection')
    runtime.json_write(base/'protocol.json',protocol)
    runtime.json_write(desktop/'experiment-configuration.json',protocol)
    jobs=[];comparisons=[];started=time.perf_counter();rates={}
    def status(stage,state='running',**extra):
        value=dict(stage=stage,status=state,pid=os.getpid(),jobs=jobs,comparisons=comparisons,
            elapsed_seconds=time.perf_counter()-started,**extra)
        runtime.json_write(base/'controller.json',value)
        runtime.json_write(desktop/'status.json',value)
        latest=(extra.get('progress') or {}).get('latest',{})
        text=['128 agents x16 environments; target:1024 updates per method, three paired seeds.',
              f'Status: {state}; stage: {stage}; current run: {extra.get("current_run", "")}',
              f'Current round: {latest.get("round", "")}; completed updates: {latest.get("optimizer_updates", "")}',
              f'Elapsed: {(time.perf_counter()-started)/60:.1f} minutes']
        if extra.get('estimated_remaining_seconds') is not None:
            text.append(f'Estimated remaining: {extra["estimated_remaining_seconds"]/60:.1f} minutes')
        (desktop/'progress.txt').write_text('\n'.join(text)+'\n')
        return value
    runtime.publish('coverage512_controlled_parity_starting',status='running',
        artifact_root=str(base),agents=128,envs=16,target_updates=1024)
    try:
        for phase,seeds,rounds in [('proof',[0],4),('timing',[0,1,2],32)]:
            for seed in seeds:
                order=['cpu','jointloom'] if seed%2==0 else ['jointloom','cpu']
                for backend in order:
                    name=f'{phase}-{backend}-n16-s{seed}';out=base/name
                    command=[str(runtime.PYTHON),str(frozen/'train_dagger.py'),
                        '--backend',backend,'--out',str(out),'--agents','128','--envs','16',
                        '--workers','8','--rounds',str(rounds),'--steps','512','--queries','128',
                        '--updates','32','--batch-size','64','--seed',str(seed),
                        '--freeze-batchnorm','--background-size','512','--background-agents','64','--online-per-batch','16',
                        '--learning-rate','0.000001','--deterministic','--independent-rngs',
                        '--expert-iterations','2000','--record-trajectory',
                        '--query-sampling','stratified','--query-time-bins','8',
                        '--replay-sampling','episode-stratified']
                    if phase=='proof':command.append('--save-round-states')
                    job=dict(name=name,status='running',command=command);jobs.append(job)
                    t=time.perf_counter()
                    with (base/(name+'.log')).open('w') as log:
                        child=subprocess.Popen(command,cwd=runtime.PAPER,stdout=log,stderr=subprocess.STDOUT)
                        while True:
                            try:code=child.wait(timeout=2);break
                            except subprocess.TimeoutExpired:
                                progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else None
                                remaining=None
                                if phase=='timing' and rates:
                                    done=(progress or {}).get('latest',{}).get('round',0)
                                    remaining=(rounds-done)*rates[backend]
                                    # Include every subsequent training job, plus heldout evaluation.
                                    jobs_left=6-sum(j['name'].startswith('timing-') for j in jobs)
                                    remaining+=jobs_left*32*sum(rates.values())/2+1800
                                status(phase,current_run=name,child_pid=child.pid,
                                    progress=progress,estimated_remaining_seconds=remaining)
                    elapsed=time.perf_counter()-t
                    job.update(status='complete' if code==0 else 'failed',returncode=code,
                        process_wall_seconds=elapsed)
                    if code:raise RuntimeError(f'{name} failed; see {base/(name+".log")}')
                    runtime.json_write(out/'controller-timing.json',dict(process_wall_seconds=elapsed))
                    job['result']=json.loads((out/'result.json').read_text())
                    rates[backend]=job['result']['online_seconds']/rounds
                    print(name,job['result']['online_seconds'],flush=True)
                result=compare_pair(base/f'{phase}-cpu-n16-s{seed}',base/f'{phase}-jointloom-n16-s{seed}',
                    include_rounds=phase=='proof')
                runtime.json_write(base/f'{phase}-s{seed}-comparison.json',result)
                comparisons.append(dict(phase=phase,seed=seed,identical=result['identical']))
                status('comparison',latest=comparisons[-1])
                print(comparisons[-1],flush=True)
                if not result['identical']:raise RuntimeError('pair differs; locate first divergence')
                if phase=='proof':
                    estimate=3*32*sum(rates.values())+6*8+1800
                    runtime.json_write(base/'forecast-after-proof.json',dict(
                        remaining_seconds=estimate,per_round_seconds=rates,
                        includes='six full training runs; ten checkpoints on shared fresh 128 tasks'))
                    shutil.copy2(base/'forecast-after-proof.json',desktop/'forecast-after-pilot.json')
        status('training_complete',state='training_complete')
        evaluate=[str(runtime.PYTHON),str(frozen/'evaluate_controlled_parity.py'),
            '--base',str(base),'--desktop',str(desktop),'--seed-base','2037000000',
            '--tasks','128','--training-envs','16','--expected-updates','1024','--agents','128',
            '--reference-run-root',str(runtime.ART/'controlled-parity-agents128-u1024')]
        status('evaluation',state='training_complete')
        with (base/'evaluation.log').open('w') as log:
            subprocess.run(evaluate,cwd=runtime.PAPER,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (base/'report.log').open('w') as log:
            subprocess.run([str(runtime.PYTHON),str(frozen/'report_full_coverage.py')],
                cwd=runtime.PAPER,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (base/'plot.log').open('w') as log:
            subprocess.run([str(runtime.PYTHON),str(frozen/'plot_controlled_parity.py'),
                '--base',str(base),'--desktop',str(desktop),'--training-envs','16',
                '--updates','1024','--loss-window','32','--pretrained-reference','--previous-reference'],cwd=runtime.PAPER,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        status('complete',state='complete')
        runtime.publish('coverage512_controlled_parity_complete',status='complete',artifact_root=str(base))
        print('scaled online parity, timing and quality evaluation complete',flush=True)
    except BaseException:
        import traceback
        error=traceback.format_exc()
        status('failed',state='failed',error=error)
        runtime.publish('coverage512_controlled_parity_failed',status='failed',artifact_root=str(base),error=error)
        raise


if __name__=='__main__':main()
