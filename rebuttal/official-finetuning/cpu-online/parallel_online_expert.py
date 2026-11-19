"""Overlap ordered GPU policy rollouts with a bounded CPU expert pool.

Policy actions are sampled in the original instance order. Expert results are
committed in that same order, before the following learning epoch begins.
"""
import official_bootstrap as bootstrap
import ast
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


@dataclass
class ObservationReference:
    directory: str
    steps: int


def initialize_worker(workers):
    allowed = sorted(os.sched_getaffinity(0))
    index = (mp.current_process()._identity[-1] - 1) % workers
    os.sched_setaffinity(0, allowed[index::workers])
    torch.set_num_threads(1)


def solve_case(payload):
    from argparse import Namespace
    from pogema import GridConfig
    from magat_plus.run_expert import get_expert_algorithm_and_config, run_expert_algorithm
    from data_store import save_case

    started = time.perf_counter()
    path = Path(payload['directory'])
    path.mkdir(parents=True, exist_ok=False)
    native = path / 'native'
    native.mkdir()
    os.chdir(native)
    args = Namespace(**payload['args'])
    config = GridConfig(**payload['grid_config'])
    algorithm, inference = get_expert_algorithm_and_config(args)
    inference.lacam_lib_path = str(bootstrap.LACAM_LIBRARY)
    actions, observations, terminated = run_expert_algorithm(
        algorithm(inference), grid_config=config, save_termination_state=True)
    solved_at = time.perf_counter()
    success = bool(all(terminated[-1]))
    result = dict(index=payload['index'], success=success, pid=os.getpid(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  expert_seconds=solved_at-started)
    if success:
        saved = save_case(path, payload['case_id'], config, actions, observations,
                          terminated, args, expanded=payload['expanded'])
        result.update(directory=str(path), steps=saved['steps'])
    result['total_seconds'] = time.perf_counter()-started
    (path/'expert-timing.json').write_text(json.dumps(result, indent=2))
    return result


def rng_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all(),
            np.random.get_state(), random.getstate())


def restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])
    np.random.set_state(state[2])
    random.setstate(state[3])


def run_round(model, device, oe_ids, train_id_max, oe_grid_configs, factory,
              seeds, args, dataset_kwargs, epoch, *, run_policy, log, workers,
              expanded, out):
    from grid_config_generator import generate_grid_config_from_env

    started = time.perf_counter()
    pending = deque()
    results = []
    rollouts = []
    states = []
    completed = 0
    cutoff = None
    discarded = 0
    out = Path(out)
    root = out/'online-data'/f'epoch-{epoch+1:02d}'
    root.mkdir(parents=True, exist_ok=False)

    def progress(phase):
        log.phase = phase
        log.state(epoch=epoch+1, oe_policy_completed=len(rollouts),
                  oe_policy_successes=sum(r['success'] for r in rollouts),
                  oe_total=len(oe_ids), oe_expert_submitted=len(results)+len(pending),
                  oe_expert_completed=completed, expert_workers=workers)

    def collect_first():
        nonlocal completed, cutoff
        index, config, future = pending.popleft()
        result = future.result()
        completed += 1
        with (out/'online-experts.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=epoch+1, **result))+'\n')
        results.append((index, config, result))
        if not result['success']:
            cutoff = index
        progress('online_expert_parallel_wait')

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(workers,)) as pool:
        for i, graph_id in enumerate(oe_ids):
            # Retain the public trainer's recursive task lookup and boundary.
            if graph_id > train_id_max:
                config = oe_grid_configs[graph_id-train_id_max]
            else:
                config = factory(seeds[graph_id], map_id=graph_id)
            log.wall_checkpoint(model)
            progress('online_expert_policy_rollout')
            success, env, observations = run_policy(model=model, device=device,
                grid_config=config, args=args, dataset_kwargs=dataset_kwargs,
                max_episodes=args.max_episode_steps)
            row = dict(phase='online_expert_policy_rollout', epoch=epoch+1,
                case_id=int(graph_id), success=bool(success),
                agent_arrival=float(np.mean(env.was_on_goal)), agents=int(env.num_agents),
                elapsed_seconds=time.perf_counter()-log.started,
                final_positions=np.asarray(env.grid.get_agents_xy()).tolist())
            rollouts.append(row)
            states.append(rng_state())
            if not success:
                correction = generate_grid_config_from_env(env)
                payload = dict(index=i, case_id=1_000_000+(epoch+1)*1000+i,
                    directory=str(root/f'{i:04d}'), grid_config=correction.dict(),
                    args=vars(args), expanded=expanded)
                pending.append((i, correction, pool.submit(solve_case, payload)))
            progress('online_expert_policy_rollout')
            print(f'Parallel OE epoch {epoch+1}: policy {i+1}/{len(oe_ids)}, '
                  f'expert completed {completed}, pending {len(pending)}', flush=True)
            # Keep memory and speculative CPU work bounded.
            while pending and (len(pending) >= workers*2 or pending[0][2].done()):
                collect_first()
                if cutoff is not None:
                    break
            if cutoff is not None:
                break
        while pending and cutoff is None:
            progress('online_expert_parallel_wait')
            log.wall_checkpoint(model)
            collect_first()
        if cutoff is not None:
            # Public code stops the round at its first unsuccessful expert.
            # Discard any speculation after that point and retain its RNG state.
            restore_rng(states[cutoff])
            for _, _, future in pending:
                future.cancel()
            discarded = len(rollouts)-cutoff-1
            rollouts = rollouts[:cutoff+1]

    accepted = [(i, cfg, r) for i, cfg, r in results
                if r['success'] and (cutoff is None or i < cutoff)]
    oe_dataset = [(ObservationReference(r['directory'], r['steps']), None, None)
                  for _, _, r in accepted]
    configs = [cfg for _, cfg, _ in accepted]
    # The storage adapter derives graph data directly from saved references.
    add_data = [None] * sum(r['steps'] for _, _, r in accepted)
    with (out/'rollouts.jsonl').open('a') as f:
        for row in rollouts:
            f.write(json.dumps(row)+'\n')
    summary = dict(epoch=epoch+1, policy_rollouts=len(rollouts),
        policy_successes=sum(r['success'] for r in rollouts),
        expert_successes=len(accepted), expert_workers=workers,
        expert_commit_indices=[i for i, _, _ in accepted],
        elapsed_seconds=time.perf_counter()-started, stopped_at_failed_expert=cutoff,
        discarded_speculative_rollouts=discarded)
    with (out/'online-rounds.jsonl').open('a') as f:
        f.write(json.dumps(summary)+'\n')
    print('Parallel OE complete: '+json.dumps(summary), flush=True)
    return oe_dataset, add_data, summary['policy_successes'], configs


def install(official, callback, epoch_callback=None):
    """Replace only online task scheduling; retain the public learning loop."""
    source_path = Path(official.__file__)
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')

    class Schedule(ast.NodeTransformer):
        replaced = 0
        def visit_For(self, node):
            if isinstance(node.target, ast.Name) and node.target.id=='epoch':
                node=self.generic_visit(node)
                if epoch_callback is not None:
                    call=ast.parse('_save_epoch_state(locals())').body[0]
                    ast.copy_location(call,node)
                    node.body.append(call)
                return node
            if (isinstance(node.target, ast.Tuple) and
                    [getattr(e, 'id', None) for e in node.target.elts]==['i','graph_id']):
                self.replaced += 1
                text = ('oe_dataset, oe_add_data, num_oe_success, _new_configs = '
                        '_parallel_online_round(model, device, oe_ids, train_id_max, '
                        'oe_grid_configs, _grid_config_generator, seeds, args, '
                        'dataset_kwargs, epoch)\noe_grid_configs.extend(_new_configs)')
                replacement = ast.parse('\n'*(node.lineno-1)+text).body
                return replacement
            return self.generic_visit(node)

    change = Schedule()
    main = change.visit(main)
    if change.replaced != 1:
        raise RuntimeError('Could not identify the public online task loop')
    official._parallel_online_round = callback
    official._save_epoch_state = epoch_callback
    module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
    exec(compile(module, str(source_path), 'exec'), official.__dict__)
