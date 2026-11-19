import torch
from expert.fixed_magat_plus_runtime import FixedMAGATPlusModel, PyGBatchBuilder
from mapf_cuda.models.checkpoints import load_magat_checkpoint
from types import SimpleNamespace


class MagatPolicy:
    def __init__(self, checkpoint, device):
        self.model=FixedMAGATPlusModel().to(device)
        load_magat_checkpoint(SimpleNamespace(model=self.model),checkpoint,device)
        self.model.eval()
        self.builder=PyGBatchBuilder()

    @torch.inference_mode()
    def act(self,data):
        return self.model(data.x,data).argmax(dim=-1).to(torch.uint8)

    @torch.inference_mode()
    def act_resident(self,sim,agents,env_batch):
        data=self.builder.build_from_stateful(sim,materialize_edges=True)
        return self.act_env_batches(data,agents,env_batch)

    @torch.inference_mode()
    def act_env_batches(self,data,agents,env_batch):
        if len(data.x)<=env_batch*agents:
            return self.act(data).reshape(-1,agents)
        actions=[]
        for start in range(0,len(data.x),env_batch*agents):
            end=min(len(data.x),start+env_batch*agents)
            part=self.builder.slice_env_aligned_batch(data,start,end,agents)
            actions.append(self.act(part))
        return torch.cat(actions).reshape(-1,agents)
