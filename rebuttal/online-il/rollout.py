"""Owned GPU observations, whole-configuration graph minibatches and PPO fields."""
import torch
from expert.fixed_magat_plus_runtime import RuntimePyGBatch


class ExpandedRolloutStore:
    def __init__(self,steps,envs,agents,state_dim,device):
        self.steps=steps;self.envs=envs;self.agents=agents;self.device=device
        base=(steps,envs,agents)
        self.x=torch.empty((*base,4,13,13),device=device)
        self.edges=torch.zeros((*base,agents),dtype=torch.bool,device=device)
        self.attr=torch.zeros((*base,agents,3),device=device)
        self.states=torch.empty((steps,envs,state_dim),device=device)
        for name in ['actions','logp','values','rewards','continuation','active']:
            dtype=torch.long if name=='actions' else torch.float32
            setattr(self,name,torch.empty(base,device=device,dtype=dtype))

    @torch.no_grad()
    def put(self,t,data,state,active,actions,logp,values,reward,continuation):
        a=self.agents
        self.x[t].copy_(data.x.reshape_as(self.x[t]));self.states[t].copy_(state)
        self.edges[t].zero_();self.attr[t].zero_()
        src,dst=data.edge_index
        env=src//a;i=src%a;j=dst%a
        self.edges[t,env,i,j]=True;self.attr[t,env,i,j]=data.edge_attr
        for name,value in [('active',active),('actions',actions),('logp',logp),('values',values),
                ('rewards',reward),('continuation',continuation)]:
            getattr(self,name)[t].copy_(value.reshape_as(getattr(self,name)[t]))

    def graph(self,ids):
        a=self.agents;b=len(ids)
        x=self.x.flatten(0,1)[ids].flatten(0,1)
        edge_mask=self.edges.flatten(0,1)[ids]
        env,src,dst=edge_mask.nonzero(as_tuple=True)
        edge_index=torch.stack((env*a+src,env*a+dst))
        attr=self.attr.flatten(0,1)[ids][env,src,dst]
        return RuntimePyGBatch(x=x,edge_index=edge_index,edge_attr=attr,
            batch=torch.arange(b,device=self.device).repeat_interleave(a),
            ptr=torch.arange(0,(b+1)*a,a,device=self.device))

    def allocated_bytes(self):
        return sum(v.numel()*v.element_size() for v in vars(self).values() if torch.is_tensor(v))
