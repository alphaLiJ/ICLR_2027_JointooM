"""Check real expert labels and changed-goal inputs, then a real CE update."""
import bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from tasks import make_tasks
from pogema_path import OriginalPogemaLifelong
from lifelong_il import LifelongILConsumer
from run_e2_il import make_learner,configure


def main(trace_dir):
    c=yaml.safe_load((trace_dir/'config.yaml').read_text());configure(c)
    tasks=make_tasks(c['map_name'],c['agents'],c['envs'],c['task_seed'])
    records=np.load(trace_dir/'records.npy')
    candidates=np.flatnonzero(np.any(records[1:,:,7]!=0,axis=1))
    stages=[0]+([int(candidates[0])+1] if len(candidates) else [])
    cpu=OriginalPogemaLifelong(tasks,prepare=False)
    original=LifelongILConsumer('pogema_original',tasks,c['device'])
    compact=LifelongILConsumer('jointloom',tasks,c['device'])
    learner=make_learner(c);learner.model.eval();rows=[]
    for seq,stage in enumerate(stages):
        raw=records[stage];cpu.load_rows(raw);arrays=cpu.build()
        original.upload(dict(raw=raw,arrays=arrays),seq);compact.upload(dict(raw=raw),seq)
        a=original.get(seq);b=compact.get(seq)
        error=float((a.x-b.x).abs().max().item());assert error<1e-6,error
        assert torch.equal(a.y,b.y) and torch.equal(a.edge_index,b.edge_index) and torch.equal(a.edge_attr,b.edge_attr)
        with torch.no_grad():la=float(learner.loss_from_batch(a));lb=float(learner.loss_from_batch(b))
        assert abs(la-lb)<1e-5,(la,lb)
        rows.append(dict(stage=stage,changed_goals=int(np.sum(raw[:,7]!=0)),max_input_error=error,
            labels_equal=True,edges_equal=True,original_eval_loss=la,compact_eval_loss=lb))
        original.release(seq);compact.release(seq)
    learner.model.train();before=next(learner.model.parameters()).detach().clone()
    loss=learner.train_step_from_batch(b);torch.cuda.synchronize()
    assert bool(torch.isfinite(loss))
    assert not torch.equal(before,next(learner.model.parameters()))
    print(json.dumps(dict(check='PASS',inputs=rows,real_training_update_loss=float(loss.item())),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True)
    main(p.parse_args().trace)
