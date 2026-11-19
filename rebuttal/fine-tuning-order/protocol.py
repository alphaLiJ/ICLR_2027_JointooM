"""Shared paths, experiment constants and the public training data schedule."""

import csv
import json
import math
import os
from pathlib import Path
import sys


WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent))
from repository_paths import REPO, ARTIFACTS, PYTHON, input_path
PAPER = REPO
MEDIA = ARTIFACTS
OFFICIAL = input_path('JOINTLOOM_MAGAT_ARTIFACTS', ARTIFACTS / 'magat-official')
REFERENCE = input_path('JOINTLOOM_MAGAT_REFERENCE', OFFICIAL / 'formal/jointloom')
ROOT = input_path('JOINTLOOM_ORDER_ARTIFACTS', ARTIFACTS / 'magat-order')
DESKTOP = input_path('JOINTLOOM_REPORT_DIR', ROOT / 'reports')
WAREHOUSE128 = input_path('JOINTLOOM_WAREHOUSE128', ARTIFACTS / 'maps/warehouse32')
LEGACY_MODULES = (
    'official_bootstrap.py', 'configuration.py', 'data_store.py',
    'gpu_inputs.py', 'train_official.py', 'parallel_online_expert.py',
)
ORDERS = ('ordered', 'window-128', 'global-shuffle')
EPOCHS = 52
BATCH_SIZE = 64
SEED = 42


def activate_official():
    os.environ['JOINTLOOM_PAPER_ROOT'] = str(PAPER)
    snapshot = WORK / 'base'
    source = snapshot if snapshot.exists() else REPO / 'rebuttal/official-finetuning/gpu'
    sys.path.insert(0, str(source))


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as lines:
        return [json.loads(line) for line in lines if line.strip()]


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_manifest():
    corpus = read_json(REFERENCE / 'input-corpus.json')
    initial = [case for case in read_json(OFFICIAL / 'dataset/index.json') if case['success']]
    training_ids = set(corpus['training_case_ids'])
    initial_graphs = sum(case['steps'] for case in initial if case['case_id'] in training_ids)
    experts = {(v['epoch'], v['index']): v for v in read_jsonl(REFERENCE / 'online-experts.jsonl')}
    rounds = []
    for cycle in read_jsonl(REFERENCE / 'online-rounds.jsonl'):
        epoch = cycle['epoch']
        if epoch >= EPOCHS:
            continue
        accepted = [experts[epoch, i] for i in cycle['expert_commit_indices']]
        cases = [read_json(Path(record['directory']) / 'result.json') for record in accepted]
        if len(cases) != cycle['expert_successes']:
            raise ValueError(f'Unexpected number of accepted corrections at epoch {epoch}')
        rounds.append(dict(available_after_epoch=epoch, cases=cases,
                           reference_rollout_successes=cycle['policy_successes']))
    active = 0
    steps_by_epoch = []
    added = {r['available_after_epoch']: r['cases'] for r in rounds}
    for epoch in range(1, EPOCHS + 1):
        steps_by_epoch.append(dict(epoch=epoch, initial_graphs=initial_graphs,
            correction_graphs=active, updates=math.ceil(initial_graphs / BATCH_SIZE) + math.ceil(active / BATCH_SIZE)))
        active += sum(case['steps'] for case in added.get(epoch, []))
    updates = sum(row['updates'] for row in steps_by_epoch)
    if updates != read_json(REFERENCE / 'result.json')['optimizer_updates']:
        raise ValueError('Replayed corpus and the reference run have different update counts')
    return dict(seed=SEED, two_training_pools=True, epochs=EPOCHS, batch_size=BATCH_SIZE,
        source_run=str(REFERENCE), corpus_split=corpus, initial_cases=initial,
        initial_training_graphs=initial_graphs, correction_rounds=rounds,
        correction_cases=sum(len(r['cases']) for r in rounds), correction_graphs=active,
        optimizer_updates=updates, per_epoch=steps_by_epoch)
