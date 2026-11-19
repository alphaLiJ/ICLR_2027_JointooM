"""A common official evaluator for ordered, windowed and global fine-tuning."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

from protocol import ORDERS, SEED, WAREHOUSE128, activate_official, read_json, read_jsonl, write_csv, write_json

activate_official()
import official_bootstrap
import numpy as np
import torch
from pogema import GridConfig
from configuration import configuration
from grid_config_generator import grid_config_generator_factory
from magat_plus.magat.agents import get_model
from magat_plus.run_model import run_model_on_grid
from magat_plus.train_imitation_learning_pyg import aux_func


def task_groups(manifest, preview):
    heldout = configuration()
    seeds = np.random.default_rng(heldout.dataset_seed).integers(10**10, size=heldout.num_samples)
    ids = manifest['corpus_split']['test_case_ids']
    warehouse = configuration()
    warehouse.num_agents = '128+128'
    warehouse.map_dir = str(WAREHOUSE128)
    for name, args, task_ids, task_seeds in (
        ('warehouse-heldout150', heldout, ids, [int(seeds[i]) for i in ids]),
        ('warehouse32-agents128', warehouse, list(range(128)), list(range(2037000000, 2037000128))),
    ):
        factory = grid_config_generator_factory(args)
        tasks = [dict(task_id=case, seed=seed, grid_config=json.loads(factory(seed, map_id=case).json()))
                 for case, seed in zip(task_ids, task_seeds)]
        yield name, args, tasks[:1] if preview else tasks


def checkpoints(training, preview):
    if preview:
        yield dict(order='ordered', selection='preview', epoch=2,
                   path=str(training/'ordered/checkpoints/last.pt'))
        return
    for order in ORDERS:
        yield dict(order=order, selection='fixed-final', epoch=52,
                   path=str(training/order/'checkpoints/last.pt'))
    for order in ORDERS:
        epochs = [row for row in read_jsonl(training/order/'epochs.jsonl') if 'validation_success_rate' in row]
        best = max(epochs, key=lambda row: (row['validation_success_rate'], -row['epoch']))
        epoch = best['epoch']
        name = 'last.pt' if epoch == 52 else f'epoch_{epoch-1}.pt'
        yield dict(order=order, selection='validation-selected', epoch=epoch,
                   validation_success_rate=best['validation_success_rate'], path=str(training/order/'checkpoints'/name))


def summary_for(setting, model, rows):
    solved = [row for row in rows if row['success']]
    return dict(setting=setting, order=model['order'], selection=model['selection'], epoch=model['epoch'],
        tasks=len(rows), successes=len(solved), success_rate=len(solved)/len(rows),
        mean_agent_arrival=float(np.mean([row['agent_arrival'] for row in rows])),
        successful_soc_mean=float(np.mean([row['soc'] for row in solved])) if solved else None,
        successful_makespan_mean=float(np.mean([row['makespan'] for row in solved])) if solved else None)


def paired_costs(instances):
    groups = {}
    for row in instances:
        key = (row['setting'], row['selection'], row['order'])
        groups.setdefault(key, {})[row['task_id']] = row
    result = []
    for setting, selection in sorted({key[:2] for key in groups}):
        for first, second in combinations(ORDERS, 2):
            left = groups.get((setting, selection, first))
            right = groups.get((setting, selection, second))
            if left is None or right is None:
                continue
            if left.keys() != right.keys():
                raise ValueError(f'Unpaired evaluation task sets for {first} versus {second}')
            left_ok = {k for k, row in left.items() if row['success']}
            right_ok = {k for k, row in right.items() if row['success']}
            common = sorted(left_ok & right_ok)
            row = dict(setting=setting, selection=selection, order_a=first, order_b=second,
                tasks=len(left), common_successes=len(common), success_a_only=len(left_ok-right_ok),
                success_b_only=len(right_ok-left_ok))
            for metric in ('soc', 'makespan'):
                for suffix, cohort in (('a', left), ('b', right)):
                    row[f'mean_{metric}_{suffix}'] = float(np.mean([cohort[k][metric] for k in common])) if common else None
            result.append(row)
    return result


def publish_report(out, summaries, paired):
    def fmt(value):
        return f'{value:.2f}' if value is not None else '—'

    lines = ['# Sampling-order comparison after MAGAT+ fine-tuning', '',
        'The three orders share raw expert data, pretrained weights and update budgets. The main comparison uses the fixed final checkpoint. All tasks use the same public MAGAT+ evaluator and per-task action seeds.', '',
        '| task | checkpoint rule | sampling order | epoch | successes/total | agent arrival rate | successful-task SoC | successful-task makespan |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        lines.append(f'| {row["setting"]} | {row["selection"]} | {row["order"]} | {row["epoch"]} | '
            f'{row["successes"]}/{row["tasks"]} | {row["mean_agent_arrival"]:.3%} | '
            f'{fmt(row["successful_soc_mean"])} | {fmt(row["successful_makespan_mean"])} |')
    if paired:
        lines += ['', 'Cost comparisons use tasks solved by both methods.', '',
            '| task | checkpoint rule | order A / B | shared successes | only A / only B | SoC A / B | makespan A / B |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: |']
        for row in paired:
            lines.append(f'| {row["setting"]} | {row["selection"]} | {row["order_a"]} / {row["order_b"]} | '
                f'{row["common_successes"]} | {row["success_a_only"]} / {row["success_b_only"]} | '
                f'{fmt(row["mean_soc_a"])} / {fmt(row["mean_soc_b"])} | '
                f'{fmt(row["mean_makespan_a"])} / {fmt(row["mean_makespan_b"])} |')
    (out/'RESULTS.md').write_text('\n'.join(lines) + '\n')


def run(cli):
    torch.set_num_threads(1)
    cli.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    groups = list(task_groups(read_json(cli.manifest), cli.preview))
    selected = list(checkpoints(cli.training, cli.preview))
    write_json(cli.out/'selected-checkpoints.json', selected)
    write_json(cli.out/'tasks.json', {name: tasks for name, _, tasks in groups})
    write_json(cli.out/'protocol.json', dict(horizon=512, sampling='probabilistic', temperature=1.0,
        shielding='naive', runner='released MAGAT+ run_model_on_grid',
        action_rng='official per-task generator seeded from the frozen GridConfig.seed',
        task_sets={name: len(tasks) for name, _, tasks in groups}, selection_order=['fixed-final', 'validation-selected']))
    device = torch.device('cuda')
    cache = {}
    summaries, instances = [], []
    for current, checkpoint in enumerate(selected, 1):
        for setting, args, tasks in groups:
            path = checkpoint['path']
            cache_key = (path, setting)
            if cache_key not in cache:
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                model, kwargs = get_model(args, device)
                model.load_state_dict(torch.load(path, map_location='cuda', weights_only=False))
                model.eval()
                rows = []
                for index, task in enumerate(tasks):
                    config = GridConfig(**task['grid_config'])
                    success, env, _ = run_model_on_grid(model, device, config, args, kwargs, aux_func=aux_func)
                    rows.append(dict(setting=setting, task_id=task['task_id'], seed=task['seed'],
                        action_seed=int(config.seed), agents=env.num_agents, success=bool(success),
                        agent_arrival=float(np.mean(env.was_on_goal)),
                        soc=float(np.sum(aux_func.costs)) if success else None,
                        makespan=int(aux_func.makespan) if success else None))
                    write_json(cli.out/'progress.json', dict(status='running', setting=setting,
                        order=checkpoint['order'], selection=checkpoint['selection'], model=current,
                        models=len(selected), completed=index+1, total=len(tasks),
                        elapsed_seconds=time.perf_counter()-started))
                cache[cache_key] = rows
                del model
            rows = cache[cache_key]
            model_info = {key: checkpoint[key] for key in ('order', 'selection', 'epoch')}
            instances.extend(dict(**model_info, **row) for row in rows)
            summaries.append(summary_for(setting, checkpoint, rows))
            pairs = paired_costs(instances)
            write_csv(cli.out/'instances.csv', instances)
            write_csv(cli.out/'summary.csv', summaries)
            write_csv(cli.out/'paired-costs.csv', pairs)
            publish_report(cli.out, summaries, pairs)
            print(json.dumps(summaries[-1]), flush=True)
    final = dict(status='complete', elapsed_seconds=time.perf_counter()-started, summaries=summaries, paired_costs=pairs)
    write_json(cli.out/'summary.json', final)
    write_json(cli.out/'progress.json', dict(status='complete', elapsed_seconds=final['elapsed_seconds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    run(parser.parse_args())
