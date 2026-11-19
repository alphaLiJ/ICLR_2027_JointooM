"""Run the public MAGAT+ learner against one order of the shared raw corpus."""

import argparse
from functools import cache
import json
import os
from pathlib import Path
import sys
import time
import traceback

from protocol import ORDERS, SEED, activate_official, read_json, write_json

activate_official()
import official_bootstrap as bootstrap
import numpy as np
import torch
from pogema import GridConfig
from configuration import official_arguments, official_parser
from data_store import Frame, PublishedGraphDataset
from parallel_online_expert import install, rng_state
from train_official import RunLog, ObservationReference, dense_references, scalars
import gpu_inputs


@cache
def raw_arrays(directory):
    path = Path(directory)
    return {key: np.load(path / f'{key}.npy', mmap_mode='r')
            for key in ('compact', 'terminated', 'grid')}


@cache
def require_recorded_occupancy(directory):
    if not read_json(Path(directory) / 'result.json')['occupancy_in_record']:
        raise ValueError(f'Missing recorded observation occupancy in {directory}')


def frames_from_cases(cases):
    return [Frame(case['directory'], step, case['case_id'])
            for case in cases for step in range(case['steps'])]


def permutation(size, order, seed, epoch, pool):
    indexes = np.arange(size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, pool]))
    if order == 'global-shuffle':
        rng.shuffle(indexes)
    elif order == 'window-128':
        for start in range(0, size, 128):
            rng.shuffle(indexes[start:start + 128])
    return indexes


class TrainingLog(RunLog):
    def __init__(self, directory, order):
        self.order = order
        self.active_epoch = 0
        self.pool = 'startup'
        self.validation_completed = 0
        super().__init__(directory, 'jointloom')

    def state(self, **extra):
        return super().state(order=self.order, active_epoch=self.active_epoch,
            pool=self.pool, validation_completed=self.validation_completed, **extra)

    def wall_checkpoint(self, model):
        # The primary checkpoint is at the shared update budget.
        pass


class OrderedLoader:
    def __init__(self, dataset, batch_size, builder, log, pool, seed):
        self.frames = dataset.frames
        self.batch_size = batch_size
        self.builder = builder
        self.log = log
        self.pool = pool
        self.seed = seed

    def __len__(self):
        return (len(self.frames) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_().item()
        if self.pool == 'initial_train':
            self.log.active_epoch += 1
            self.log.validation_completed = 0
        self.log.pool = self.pool
        order = 'ordered' if self.pool == 'validation_accuracy' else self.log.order
        pool_seed = 1 if self.pool == 'correction_train' else 0
        indexes = permutation(len(self.frames), order, self.seed, self.log.active_epoch, pool_seed)
        with (self.log.out / 'sampling.jsonl').open('a') as log:
            log.write(json.dumps(dict(epoch=self.log.active_epoch, pool=self.pool,
                order=order, frames=len(indexes), batches=len(self),
                first_frames=[[self.frames[i].case_id, self.frames[i].step] for i in indexes[:8]])) + '\n')
        for offset in range(0, len(indexes), self.batch_size):
            yield self.builder.build([self.frames[i] for i in indexes[offset:offset + self.batch_size]])


def probe_global_inputs(manifest, out):
    train_ids = set(manifest['corpus_split']['training_case_ids'])
    frames = frames_from_cases([case for case in manifest['initial_cases'] if case['case_id'] in train_ids])
    indexes = permutation(len(frames), 'global-shuffle', SEED, 1, 0)[:512]
    builder = gpu_inputs.JointLoomInputs()
    start = time.perf_counter()
    for offset in range(0, len(indexes), 64):
        selection = [frames[i] for i in indexes[offset:offset + 64]]
        batch = builder.build(selection)
        if len(batch.ptr) - 1 != len(selection) or not bool(torch.isfinite(batch.x).all()):
            raise RuntimeError('Mixed-agent global-shuffle input construction failed')
    torch.cuda.synchronize()
    write_json(out / 'global-input-probe.json', dict(graphs=len(indexes),
        distinct_cases=len({frames[i].case_id for i in indexes}), goal_rebuilds=builder.goal_rebuilds,
        elapsed_seconds=time.perf_counter() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def run(cli):
    os.chdir(bootstrap.UPSTREAM)
    import magat_plus.train_imitation_learning_pyg as official
    import magat_plus.run_model as run_module

    cli.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = read_json(cli.manifest)
    initial = manifest['initial_cases']
    rounds = {round_info['available_after_epoch']: round_info for round_info in manifest['correction_rounds']}
    if cli.preview:
        initial = initial[:24]
        first = rounds[4]
        rounds = {1: dict(cases=first['cases'][:2], reference_rollout_successes=0)}
    gpu_inputs.arrays = raw_arrays
    gpu_inputs.ensure_occupancy_context = require_recorded_occupancy
    log = TrainingLog(cli.out, cli.order)
    if cli.preview and cli.order == 'global-shuffle':
        probe_global_inputs(manifest, cli.out)

    run_arguments = official_arguments(cli.out / 'checkpoints')
    run_arguments += ['--initial_val_size', '150', '--threshold_val_success_rate', '1.1']
    if cli.preview:
        run_arguments += ['--num_samples', str(len(initial)), '--num_epochs', '2',
                          '--validation_every_epochs', '1', '--initial_val_size', '2']
    args = official_parser().parse_args(run_arguments)
    write_json(cli.out / 'official-configuration.json', dict(vars(args), sampling_order=cli.order,
        expert_source='archived raw correction trajectories at their original epochs'))

    frames = frames_from_cases(initial)
    builder = gpu_inputs.JointLoomInputs()
    run_module.get_runtime_data_generator = lambda **kw: gpu_inputs.RuntimeInputs(**kw, builder=builder)

    def load_dataset(_funcs, name, _args):
        if name == 'processed_dataset':
            return dense_references(frames)
        if name == 'positions':
            return list(frames)
        if name == 'additional_data':
            return [None] * len(frames)
        raise ValueError(name)

    def new_graphs(dataset, **kwargs):
        references = []
        for obs, _, _ in dataset:
            row = read_json(Path(obs.directory) / 'result.json')
            references.extend(frames_from_cases([row]))
        return dense_references(references, with_positions=kwargs.get('use_edge_attr', False))

    loader_calls = 0

    def create_loader(dataset, batch_size):
        nonlocal loader_calls
        pool = ('initial_train', 'validation_accuracy')[loader_calls] if loader_calls < 2 else 'correction_train'
        loader_calls += 1
        return OrderedLoader(dataset, batch_size, builder, log, pool, args.model_seed)

    def replay_round(_model, _device, _ids, _train_max, _configs, _factory, _seeds, _args, _kwargs, epoch):
        cycle = rounds.get(epoch + 1, dict(cases=[], reference_rollout_successes=0))
        cases = cycle['cases']
        dataset = [(ObservationReference(case['directory'], case['steps']), None, None) for case in cases]
        configs = [GridConfig(**read_json(Path(case['directory']) / 'grid_config.json')) for case in cases]
        graphs = sum(case['steps'] for case in cases)
        row = dict(epoch=epoch + 1, archived_corrections=len(cases), added_graphs=graphs)
        with (cli.out / 'replayed-corrections.jsonl').open('a') as log_file:
            log_file.write(json.dumps(row) + '\n')
        print('Replaying archived corrections', row, flush=True)
        return dataset, [None] * graphs, cycle['reference_rollout_successes'], configs

    original_loss = official.get_loss_function

    def make_loss(settings):
        loss = original_loss(settings)
        accuracy = loss.get_accuracies

        def record(out, data, model, split='train'):
            result = accuracy(out, data, model, split)
            if split == 'train':
                log.record_update(data, model)
            return result

        loss.get_accuracies = record
        return loss

    original_validation = official.run_model_on_grid

    def validate(*a, **kw):
        log.phase = 'validation_rollout'
        if log.validation_completed % 10 == 0:
            log.state()
        answer = original_validation(*a, **kw)
        log.validation_completed += 1
        return answer

    def save_epoch(context):
        log.completed_epochs = int(context['epoch']) + 1
        log.file.flush()
        row = dict(epoch=log.completed_epochs, optimizer_updates=log.updates, learner_graphs=log.graphs,
            validation_tasks=log.validation_completed, elapsed_seconds=time.perf_counter() - log.started,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(), **scalars(context['results']))
        with (cli.out / 'epochs.jsonl').open('a') as epochs:
            epochs.write(json.dumps(row) + '\n')
        scheduler = context['lr_scheduler']
        temporary = cli.out / 'training-state.tmp'
        torch.save(dict(completed_epochs=log.completed_epochs, optimizer_updates=log.updates,
            model=context['model'].state_dict(), optimizer=context['optimizer'].state_dict(),
            scheduler=scheduler.scheduler.state_dict(), rng=rng_state()), temporary)
        temporary.replace(cli.out / 'training-state.pt')
        log.state(latest_epoch=row)

    official.load_dataset = load_dataset
    official.MAPFGraphDataset = PublishedGraphDataset
    official.DataLoader = create_loader
    official.generate_graph_dataset = new_graphs
    official.get_loss_function = make_loss
    official.run_model_on_grid = validate
    install(official, replay_round, save_epoch)
    sys.argv = [official.__file__] + run_arguments
    try:
        official.main()
        if not cli.preview and (log.completed_epochs, log.updates) != (manifest['epochs'], manifest['optimizer_updates']):
            raise ValueError('Training ended at an unexpected epoch or update budget')
        result = log.state()
        result.update(status='complete', preview=cli.preview, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      goal_rebuilds=builder.goal_rebuilds)
    except Exception:
        result = log.state()
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        log.file.close()
        write_json(cli.out / 'result.json', result)
        write_json(cli.out / 'progress.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--order', choices=ORDERS, required=True)
    parser.add_argument('--preview', action='store_true')
    run(parser.parse_args())
