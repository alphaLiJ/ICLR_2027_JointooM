"""Bounded-memory storage of official expert trajectories and exact PyG inputs."""
import official_bootstrap as bootstrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class Frame:
    directory: str
    step: int
    case_id: int


@lru_cache(maxsize=4)
def arrays(directory):
    path = Path(directory)
    return {f.stem: np.load(f, mmap_mode='r') for f in path.glob('*.npy')}


def cpu_frame(frame):
    values = arrays(frame.directory)
    t = frame.step
    lo, hi = map(int, values['edge_ptr'][t:t+2])
    return Data(x=torch.from_numpy(values['x'][t]),
        edge_index=torch.from_numpy(values['edge_index'][:,lo:hi]),
        edge_attr=torch.from_numpy(values['edge_attr'][lo:hi]),
        edge_weight=torch.from_numpy(values['edge_weight'][lo:hi]),
        y=torch.from_numpy(values['compact'][t,:,6].astype(np.int64)),
        terminated=torch.from_numpy(values['terminated'][t]))


def save_case(path, case_id, config, actions, observations, terminated, args, expanded=True):
    """Uses the published converters, including unclipped fine-tuning cost-to-go."""
    from magat_plus.convert_to_imitation_dataset import generate_graph_dataset
    from magat_plus.generate_additional_data import generate_additional_data
    from magat_plus.imitation_dataset_pyg import MAPFGraphDataset
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    steps, agents = len(actions), config.num_agents
    compact = np.zeros((steps, agents, 8), dtype=np.uint16)
    compact[:,:,1] = np.arange(agents)[None,:]
    for t, obs in enumerate(observations):
        compact[t,:,2:4] = np.asarray([o['global_xy'] for o in obs])
        compact[t,:,4:6] = np.asarray([o['global_target_xy'] for o in obs])
        compact[t,:,6] = np.asarray(actions[t])
        # Pogema1.3's stored occupancy may omit a current agent after a chain
        # move. Preserve that observable context, rather than changing the
        # official environment. Bit0=goal refresh, bit1=occupancy presence.
        compact[t,:,7] = np.asarray([o['agents'][args.obs_radius,args.obs_radius]>0 for o in obs],dtype=np.uint16)*2
    compact[0,:,7] |= 1
    np.save(path/'compact.npy', compact)
    np.save(path/'terminated.npy', np.asarray(terminated, dtype=bool))
    np.save(path/'grid.npy', np.asarray(config.map, dtype=np.uint8))
    (path/'grid_config.json').write_text(config.json())
    if expanded:
        graph = generate_graph_dataset([(observations, actions, terminated)],
            comm_radius=args.comm_radius, obs_radius=args.obs_radius, num_samples=1,
            save_termination_state=True, use_edge_attr=True, print_prefix=None,
            num_neighbour_cutoff=args.num_neighbour_cutoff,
            neighbour_cutoff_method=args.neighbour_cutoff_method,
            distance_metric=args.distance_metric, random_edge_probs=args.random_edge_probs,
            stack_with_np=False)
        additional = generate_additional_data(config, actions,
            cost_to_go=True, normalized_cost_to_go=True, clamp_value=args.clamp_cost_to_go)
        dataset = MAPFGraphDataset(graph, use_edge_attr=True,
            use_edge_attr_for_messages='positions+manhattan', additional_data=additional,
            additional_data_idx=[0,None,None])
        x = np.lib.format.open_memmap(path/'x.npy', mode='w+', dtype=np.float32,
            shape=(steps,agents,4,13,13))
        edges,attrs,weights,ptr = [],[],[],[0]
        for t in range(steps):
            data = dataset[t]
            x[t] = data.x.numpy()
            edges.append(data.edge_index.numpy())
            attrs.append(data.edge_attr.numpy())
            weights.append(data.edge_weight.numpy())
            ptr.append(ptr[-1]+data.edge_index.shape[1])
        x.flush()
        np.save(path/'edge_index.npy', np.concatenate(edges,axis=1))
        np.save(path/'edge_attr.npy', np.concatenate(attrs,axis=0))
        np.save(path/'edge_weight.npy', np.concatenate(weights,axis=0))
        np.save(path/'edge_ptr.npy', np.asarray(ptr,dtype=np.int64))
    result = dict(case_id=int(case_id), agents=int(agents), steps=steps,
        success=bool(all(terminated[-1])), seed=int(config.seed), expanded=expanded,
        directory=str(path),occupancy_in_record=True)
    (path/'result.json').write_text(json.dumps(result,indent=2))
    return result


_context_ready=set()


def ensure_occupancy_context(directory):
    """Upgrade already-prepared cases from their exact stored official inputs."""
    if directory in _context_ready:return
    path=Path(directory)
    metadata=json.loads((path/'result.json').read_text())
    if metadata.get('occupancy_in_record'):
        _context_ready.add(directory);return
    x=np.load(path/'x.npy',mmap_mode='r')
    compact=np.load(path/'compact.npy',mmap_mode='r+')
    compact[:,:,7]=(compact[:,:,7]&1)+(x[:,:,1,6,6]>0).astype(np.uint16)*2
    compact.flush()
    metadata['occupancy_in_record']=True
    (path/'result.json').write_text(json.dumps(metadata,indent=2))
    _context_ready.add(directory)


def case_frames(path, case_id=None):
    result = json.loads((Path(path)/'result.json').read_text())
    if case_id is None:
        case_id = result['case_id']
    return [Frame(str(path), t, case_id) for t in range(result['steps'])]


class PublishedGraphDataset(torch.utils.data.Dataset):
    """Same graph samples; expanded or compact representation is a loader choice."""
    def __init__(self, dense_dataset, **kwargs):
        self.frames = list(dense_dataset[0])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return cpu_frame(self.frames[index])

