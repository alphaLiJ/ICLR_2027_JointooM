"""Full-horizon coverage: direct learning equality, speed and prior-policy comparison."""
import runtime
import csv
import json
import shutil
import statistics as st


def main():
    base=runtime.ART/'coverage512-agents128-u1024'
    desktop=runtime.DESKTOP/'agents128-horizon512-stratified-u1024'
    evaluation=json.loads((base/'evaluation/summary.json').read_text())
    proof=json.loads((base/'proof-s0-comparison.json').read_text())
    assert proof['identical'] and evaluation['status']=='complete' and evaluation['all_pairs_identical']
    rows=[];checks=[];coverage=[]
    for seed in (0,1,2):
        paths={b:base/f'timing-{b}-n16-s{seed}' for b in ('cpu','jointloom')}
        runs={b:json.loads((p/'result.json').read_text()) for b,p in paths.items()}
        times={b:json.loads((p/'controller-timing.json').read_text())['process_wall_seconds'] for b,p in paths.items()}
        comparison=json.loads((base/f'timing-s{seed}-comparison.json').read_text())
        assert comparison['identical'];checks.append(comparison)
        cv={b:json.loads((p/'coverage.json').read_text()) for b,p in paths.items()}
        assert cv['cpu']==cv['jointloom'];coverage.append(dict(seed=seed,**cv['cpu']))
        a,b=runs['cpu'],runs['jointloom']
        assert a['optimizer_updates']==b['optimizer_updates']==1024
        assert a['parameter_max_change']==b['parameter_max_change'] and a['parameter_max_change']>0
        rows.append(dict(seed=seed,cpu_online_seconds=a['online_seconds'],jointloom_online_seconds=b['online_seconds'],
            online_speedup=a['online_seconds']/b['online_seconds'],cpu_process_seconds=times['cpu'],
            jointloom_process_seconds=times['jointloom'],process_speedup=times['cpu']/times['jointloom'],
            cpu_labels=a['labels'],jointloom_labels=b['labels'],updates=a['optimizer_updates'],
            agent_transitions=a['agent_transitions'],learner_agent_samples=a['learner_agent_samples'],
            parameter_max_change=a['parameter_max_change'],cpu_h2d_bytes=a['h2d_bytes'],
            jointloom_h2d_bytes=b['h2d_bytes'],payload_ratio=a['h2d_bytes']/b['h2d_bytes'],
            cpu_cuda_peak_allocated_gib=a['cuda_peak_allocated_bytes']/2**30,
            jointloom_cuda_peak_allocated_gib=b['cuda_peak_allocated_bytes']/2**30))
    by_name={v['checkpoint']:v for v in evaluation['summaries']}
    reference=by_name['official-pretrained'];changes=[]
    for seed in (0,1,2):
        old=by_name[f'reference256-cpu-n16-s{seed}'];new=by_name[f'timing-cpu-n16-s{seed}']
        changes.append(dict(seed=seed,previous256_successes=old['successes'],
            current512_successes=new['successes'],tasks=128,
            success_rate_change_pp=(new['success_rate']-old['success_rate'])*100))
    speed=st.median(r['online_speedup'] for r in rows);cold=st.median(r['process_speedup'] for r in rows)
    summary=dict(status='complete',envs=16,agents=128,updates=1024,collection_steps=512,
        queries_per_round=128,all_learning_and_quality_pairs_identical=True,
        online_paired_median_speedup=speed,process_paired_median_speedup=cold,
        mean_cpu_online_seconds=st.mean(r['cpu_online_seconds'] for r in rows),
        mean_jointloom_online_seconds=st.mean(r['jointloom_online_seconds'] for r in rows),
        mean_input_payload_ratio=st.mean(r['payload_ratio'] for r in rows),rows=rows,evaluation=evaluation,
        coverage=coverage,coverage_effect=changes,pretrained_successes=reference['successes'],
        scope='same deterministic learning protocol; full 512-step collection and stratified queries/replay')
    runtime.json_write(base/'summary.json',summary);runtime.json_write(desktop/'summary.json',summary)
    with (base/'timing.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 128 agents: full 512-step collection and stratified sampling','',
        f'For all three CPU/JointLoom pairs, final model parameters, BN state, Adam state, update records, states, actions and expert results match exactly. Median paired online speedup: **{speed:.3f}x**; full-process speedup: **{cold:.3f}x**.','',
        '## Configuration','',
        '128 agents x16 environments; official 32x32 warehouse with 416 traversable cells; released 195.pt initialization. Each round collects 512 consecutive steps in eight 64-step segments, with one expert query per environment per segment (128 queries per round). Online replay is stratified by actual episode phase.',
        '32 rounds x32 updates =1024 updates per method, with 33,554,432 agent transitions, 4096 queries and 5,242,880 learner agent samples. Each batch contains 64 graphs: 48 historical 64-agent graphs and 16 online 128-agent graphs; the historical buffer contains 512 graphs.',
        'Learning rate 1e-6, original Adam/weight decay, frozen BN running statistics, first-solution LaCAM with 2000 search iterations per query and shared deterministic FP32 settings follow the previous run. The CPU path retains parallelism, caching, pinned transfers and reuse of already constructed GPU replay graphs. Both use CPU0-15, eight environment/input workers, four expert processes and the same RTX5080.','',
        '## Equality and wall-clock timing','',
        f'The three formal pairs directly compare {sum(c["actions"]["actions"] for c in checks):,} actions, {sum(c["queried_states"] for c in checks):,} queries, 1024 updates per run and the final model/Adam state. Pilot measurements are listed separately from formal timing.','',
        '| seed | CPU online/s | JointLoom online/s | online speedup | CPU full process/s | JointLoom full process/s | full-process speedup | valid labels CPU/JL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["cpu_online_seconds"]:.2f} | {r["jointloom_online_seconds"]:.2f} | {r["online_speedup"]:.3f}× | {r["cpu_process_seconds"]:.2f} | {r["jointloom_process_seconds"]:.2f} | {r["process_speedup"]:.3f}× | {r["cpu_labels"]}/{r["jointloom_labels"]} |')
    lines+=['','Online time includes formal resets, inference/interaction, live expert calls and waiting, minibatch assembly and updates. Full-process time runs from subprocess launch to exit. Experts overlap with collection; formal timing does not save a model every round.','',
        '## Observed coverage','',
        '| seed | queries in eight collection segments | queries at episode step >=256 | queries with >=75% agents arrived | valid labels by episode phase | learner online graphs by phase |',
        '|---|---|---:|---:|---|---|']
    for c in coverage:
        lines.append(f'| {c["seed"]} | {c["queries_by_time_bin"]} | {c["queries_late_episode"]} | {c["queries_near_completion"]} | {c["labels_by_episode_bin"]} | {c["learner_online_graphs_by_episode_bin"]} |')
    lines+=['','Collection time and actual episode step are recorded separately. Early states after an automatic task reset belong to the new episode. Strata with insufficient valid labels are filled from other valid states; the table reports actual learner consumption.','',
        '## Coverage comparison on shared new tasks','',
        'Task seeds2037000000-2037000127, 128 agents, 512 steps. Previous 256-step models and current models are reevaluated on these same new tasks. The final checkpoint is fixed; test results do not select the model.','',
        f'Pretrained reference: {reference["successes"]}/128 ({reference["success_rate"]*100:.2f}%).','',
        '| seed | previous 256-step model successes/128 | new 512-step model CPU/JL successes/128 | success-rate change/percentage points |',
        '|---|---:|---:|---:|']
    for c in changes:
        n=c['current512_successes']
        lines.append(f'| {c["seed"]} | {c["previous256_successes"]} | {n}/{n} | {c["success_rate_change_pp"]:+.2f} |')
    lines+=['','## All task metrics','',
        '| model | successes/128 | agent arrival rate | successful-task SoC | successful-task makespan |',
        '|---|---:|---:|---:|---:|']
    for e in evaluation['summaries']:
        soc='—' if e['soc_mean'] is None else f'{e["soc_mean"]:.2f}'
        span='—' if e['makespan_mean'] is None else f'{e["makespan_mean"]:.2f}'
        lines.append(f'| {e["checkpoint"]} | {e["successes"]} | {e["mean_agent_arrival"]*100:.3f}% | {soc} | {span} |')
    lines+=['','Within each current CPU/JL pair, per-task success, arrival rate, SoC, makespan and complete action sequences match exactly. Cost means across previous and pretrained models use different successful subsets and do not directly measure cost improvement.',
        'The shared-task comparison reports the combined effect of longer collection, stratified queries and stratified replay.','',
        f'Raw artifacts: {base}',f'Executed source: {base/"execution-source"}']
    report='\n'.join(lines)+'\n'
    (base/'RESULTS.md').write_text(report);(desktop/'RESULTS.md').write_text(report)
    for name in ['protocol.json','timing.csv','forecast-after-proof.json','proof-s0-comparison.json',
                 'timing-s0-comparison.json','timing-s1-comparison.json','timing-s2-comparison.json']:
        shutil.copy2(base/name,desktop/name)
    for name in ['summary.csv','instances.csv']:
        shutil.copy2(base/'evaluation'/name,desktop/('independent-task-'+name))
    runtime.json_write(desktop/'sampling-coverage.json',coverage)
    print(dict(online_speedup=speed,process_speedup=cold,coverage_effect=changes,identical=True),flush=True)


if __name__=='__main__':main()
