import bootstrap
import torch
from torch import nn
from configuration import SPEC


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_block_core = nn.Sequential(nn.ReLU(), nn.Conv2d(32,32,3,padding=1),
                                           nn.ReLU(), nn.Conv2d(32,32,3,padding=1))

    def forward(self, h):
        # Released graph adds the preactivated skip, not the raw block input.
        x = self.res_block_core[0](h)
        return x + self.res_block_core[3](self.res_block_core[2](self.res_block_core[1](x)))


class CostTracer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.conv_head = nn.Sequential(nn.Conv2d(2,32,3,padding=1), Residual(), Residual(), nn.ReLU())
        self.encoder.extra_linear = nn.Sequential(nn.Linear(3872,32), nn.ReLU())
        self.critic_linear = nn.Linear(32,1)
        self.action_parameterization = nn.Module()
        self.action_parameterization.distribution_linear = nn.Linear(32,5)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=SPEC['model']['initialization_gain'])
                nn.init.zeros_(module.bias)

    def forward(self, obs):
        h = self.encoder.conv_head(obs).flatten(1)
        h = self.encoder.extra_linear(h)
        return self.action_parameterization.distribution_linear(h), self.critic_linear(h).squeeze(-1)

    def load_released(self):
        import onnx
        from onnx import numpy_helper
        weights = onnx.load(str(bootstrap.REFERENCE / 'cost-tracer.onnx'))
        self.load_state_dict({v.name: torch.from_numpy(numpy_helper.to_array(v).copy()) for v in weights.graph.initializer})


def configure_torch(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
