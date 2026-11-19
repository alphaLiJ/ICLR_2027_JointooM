"""Actual cross-episode shuffled MAGAT views and a real optimizer step."""
import bootstrap
from repository_paths import ARTIFACTS, input_path
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from config_sampler import ORDERS,configuration_order
from order_inputs import ConfigurationCorpus,CompactReplay,CachedHostBuilder
from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,default=input_path('JOINTLOOM_RAW_CORPUS', ARTIFACTS/'legacy/jointloom-training-formal-20260920-v2/inputs/magat-s42'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,set(range(16)));torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # The partial tail must be consumed exactly once, including each map's quota.
    for order in ORDERS:
        plan=configuration_order(513,4,order,99)
        assert all(np.array_equal(np.sort(plan[:,e]),np.arange(513)) for e in range(4))
        if order=='window_shuffle_128':
            assert np.array_equal(plan//128,np.broadcast_to(np.arange(513)[:,None]//128,plan.shape))
    corpus=ConfigurationCorpus(args.corpus)
    rows=[]
    for order in ORDERS:
        plan=configuration_order(corpus.stages,corpus.envs,order,42)
        host=CachedHostBuilder('magat',corpus.grids,corpus.agents)
        compact=CompactReplay(corpus)
        maximum=0.;loss_error=0.
        torch.manual_seed(42);learner=MAGATRuntimeAdapter(device='cuda:0',lr_scheduler='cosine-annealing',scheduler_total_steps=100000)
        learner.model.eval()
        for sequence in range(8):
            raw=corpus.stage(plan[sequence]);prepared=host.build(raw)
            compact.upload(raw,sequence);data=compact.get(sequence)
            torch.cuda.synchronize()
            x=data.x.cpu().numpy();error=float(np.abs(prepared.arrays['x']-x).max());maximum=max(maximum,error)
            assert error<1e-6,(order,sequence,error)
            from expert.fixed_magat_plus_runtime import PyGBatchBuilder
            edge,attr=PyGBatchBuilder._materialize_valid_edges(data.edge_index_storage,data.edge_attr_storage,data.num_edges)
            assert np.array_equal(prepared.arrays['edge_index'],edge.cpu().numpy())
            assert np.array_equal(prepared.arrays['edge_attr'],attr.cpu().numpy())
            assert np.array_equal(prepared.arrays['y'],data.y.cpu().numpy())
            if sequence==0:
                with torch.no_grad():
                    a=learner.loss_from_batch(prepared.to_device('cuda:0'))
                    b=learner.loss_from_batch(data)
                loss_error=float((a-b).abs());assert loss_error<1e-6
                before=next(learner.model.parameters()).detach().clone()
                learner.model.train();loss=learner.train_step_from_batch(data);learner.model.eval()
                assert torch.isfinite(loss)
                torch.cuda.synchronize()
                assert not torch.equal(before,next(learner.model.parameters()))
            compact.release(sequence)
        rows.append(dict(order=order,real_stages_checked=8,max_input_error=maximum,
            first_eval_loss_error=loss_error,labels_edges_equal=True,actual_update=True,
            gpu_cache_refreshes=compact.refreshes,host_cache_refreshes=host.cache_rebuilds))
        print(json.dumps(rows[-1]),flush=True)
        del learner,compact,host
    (args.out/'result.json').write_text(json.dumps(dict(corpus=str(args.corpus),checks=rows,
        sampler_tail='513 stages, each complete configuration consumed once; window boundary retained'),indent=2))


if __name__=='__main__':main()
