import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from expert._lagat_imports import ensure_lagat_on_path, repo_root_from_module_file


PROJECT_ROOT = repo_root_from_module_file(__file__)
MAGAT_PLUS_PARENT = ensure_lagat_on_path(PROJECT_ROOT)


OBS_RADIUS = 5
OBS_DIAM = 2 * OBS_RADIUS + 3
PYG_NUM_CHANNELS = 4
MAGAT_EMBEDDING_SIZE = 128
MAGAT_NUM_GNN_LAYERS = 3
MAGAT_NUM_ATTENTION_HEADS = 1
MAGAT_EDGE_RAW_DIM = 3
MAGAT_NUM_ACTIONS = 5
MAGAT_LR = 1e-3
MAGAT_LR_END = 1e-6
MAGAT_WEIGHT_DECAY = 1e-5


def _get_gnn_module():
    # Defer the PyG-dependent MAGAT+ import until we actually construct the model.
    from magat_plus.magat.model.model_selection import get_gnn_module

    return get_gnn_module


class RuntimePyGBatch:
    """Minimal PyG-style batch object for the fixed MAGAT+ runtime."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return getattr(self, key)


class PyGBatchBuilder:
    """Build GPU-resident fixed-shape MAGAT+ input batches from simulator outputs."""

    def __init__(self, obs_diam: int = OBS_DIAM, num_channels: int = PYG_NUM_CHANNELS):
        self.obs_diam = obs_diam
        self.num_channels = num_channels

    @staticmethod
    def _materialize_valid_edges(edge_index_storage, edge_attr_storage, num_edges):
        total_capacity = edge_attr_storage.shape[0]
        if total_capacity == 0:
            return edge_index_storage[:, :0], edge_attr_storage[:0]

        valid_mask = torch.arange(
            total_capacity, device=edge_attr_storage.device, dtype=num_edges.dtype
        ) < num_edges.reshape(())
        return edge_index_storage[:, valid_mask], edge_attr_storage[valid_mask]

    @staticmethod
    def _materialize_pyg_inputs(simulator) -> None:
        simulator.materialize_pyg_inputs()

    @staticmethod
    def _arrived_from_raw_batch(raw_batch: torch.Tensor) -> torch.Tensor:
        return (raw_batch[:, 2] == raw_batch[:, 4]) & (raw_batch[:, 3] == raw_batch[:, 5])

    @staticmethod
    def _arrived_from_stateful(simulator, num_nodes: int) -> torch.Tensor:
        cur_x = getattr(simulator, "cur_x", None)
        cur_y = getattr(simulator, "cur_y", None)
        goal_x = getattr(simulator, "goal_x", None)
        goal_y = getattr(simulator, "goal_y", None)
        if cur_x is None or cur_y is None or goal_x is None or goal_y is None:
            return torch.zeros(num_nodes, dtype=torch.bool, device=simulator.pyg_x.device)
        return (cur_x.reshape(-1)[:num_nodes] == goal_x.reshape(-1)[:num_nodes]) & (
            cur_y.reshape(-1)[:num_nodes] == goal_y.reshape(-1)[:num_nodes]
        )

    def build(
        self,
        simulator,
        raw_batch: torch.Tensor,
        materialize_edges: bool = True,
    ) -> RuntimePyGBatch:
        self._materialize_pyg_inputs(simulator)
        num_nodes = raw_batch.shape[0]
        x = simulator.pyg_x[:num_nodes].view(num_nodes, self.num_channels, self.obs_diam, self.obs_diam)
        y = raw_batch[:, 6].to(dtype=torch.long)
        terminated = torch.zeros(num_nodes, dtype=torch.bool, device=raw_batch.device)
        arrived = self._arrived_from_raw_batch(raw_batch)
        edge_index_storage = simulator.pyg_edge_index_storage
        edge_attr_storage = simulator.pyg_edge_attr_storage
        num_edges = simulator.pyg_num_edges
        edge_index = edge_attr = None
        if materialize_edges:
            edge_index, edge_attr = self._materialize_valid_edges(
                edge_index_storage, edge_attr_storage, num_edges
            )
        return RuntimePyGBatch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_index_storage=edge_index_storage,
            edge_attr_storage=edge_attr_storage,
            num_edges=num_edges,
            batch=simulator.pyg_batch,
            ptr=simulator.pyg_ptr,
            y=y,
            terminated=terminated,
            arrived=arrived,
        )

    @staticmethod
    def slice_env_aligned_batch(
        data: RuntimePyGBatch,
        start_node: int,
        end_node: int,
        agents_per_env: int,
    ) -> RuntimePyGBatch:
        if start_node < 0 or end_node < start_node:
            raise ValueError(f"invalid node slice: start={start_node}, end={end_node}")
        if agents_per_env <= 0:
            raise ValueError(f"agents_per_env must be positive, got {agents_per_env}")
        if start_node % agents_per_env != 0 or end_node % agents_per_env != 0:
            raise ValueError(
                "partial training batches must align to whole environments: "
                f"start={start_node}, end={end_node}, agents_per_env={agents_per_env}"
            )

        x = data.x[start_node:end_node]
        y = data.y[start_node:end_node]
        terminated = data.terminated[start_node:end_node]
        arrived = data.arrived[start_node:end_node]

        num_sub_envs = (end_node - start_node) // agents_per_env
        device = x.device
        batch = torch.arange(num_sub_envs, device=device, dtype=torch.int64).repeat_interleave(agents_per_env)
        ptr = torch.arange(
            0,
            (num_sub_envs + 1) * agents_per_env,
            step=agents_per_env,
            device=device,
            dtype=torch.int64,
        )

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if edge_index is None or edge_attr is None:
            edge_index, edge_attr = PyGBatchBuilder._materialize_valid_edges(
                data.edge_index_storage,
                data.edge_attr_storage,
                data.num_edges,
            )

        if edge_index is not None:
            edge_mask = (
                (edge_index[0] >= start_node)
                & (edge_index[0] < end_node)
                & (edge_index[1] >= start_node)
                & (edge_index[1] < end_node)
            )
            edge_index = edge_index[:, edge_mask] - start_node
            edge_attr = edge_attr[edge_mask]

        num_edges = torch.tensor(
            [0 if edge_index is None else edge_index.shape[1]],
            device=device,
            dtype=torch.int64,
        )

        return RuntimePyGBatch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_index_storage=edge_index,
            edge_attr_storage=edge_attr,
            num_edges=num_edges,
            batch=batch,
            ptr=ptr,
            y=y,
            terminated=terminated,
            arrived=arrived,
        )

    def view_stateful_outputs(
        self, simulator, materialize_edges: bool = True
    ) -> RuntimePyGBatch:
        """View already-built resident outputs without launching builder kernels."""

        num_nodes = simulator.pyg_x.shape[0]
        x = simulator.pyg_x.view(num_nodes, self.num_channels, self.obs_diam, self.obs_diam)
        y = simulator.actions.reshape(-1).to(dtype=torch.long)
        terminated = torch.zeros(num_nodes, dtype=torch.bool, device=simulator.pyg_x.device)
        arrived = self._arrived_from_stateful(simulator, num_nodes)
        edge_index_storage = simulator.pyg_edge_index_storage
        edge_attr_storage = simulator.pyg_edge_attr_storage
        num_edges = simulator.pyg_num_edges
        edge_index = edge_attr = None
        if materialize_edges:
            edge_index, edge_attr = self._materialize_valid_edges(
                edge_index_storage, edge_attr_storage, num_edges
            )
        return RuntimePyGBatch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_index_storage=edge_index_storage,
            edge_attr_storage=edge_attr_storage,
            num_edges=num_edges,
            batch=simulator.pyg_batch,
            ptr=simulator.pyg_ptr,
            y=y,
            terminated=terminated,
            arrived=arrived,
        )

    def build_from_stateful(self, simulator, materialize_edges: bool = True) -> RuntimePyGBatch:
        self._materialize_pyg_inputs(simulator)
        return self.view_stateful_outputs(
            simulator,
            materialize_edges=materialize_edges,
        )


def build_fixed_magat_args() -> SimpleNamespace:
    """Fixed MAGAT+ config pinned to the target training-script version."""
    return SimpleNamespace(
        obs_radius=OBS_RADIUS,
        embedding_size=MAGAT_EMBEDDING_SIZE,
        num_gnn_layers=MAGAT_NUM_GNN_LAYERS,
        num_attention_heads=MAGAT_NUM_ATTENTION_HEADS,
        attention_mode="MAGAT_multiplicative",
        edge_dim=None,
        model_residuals="all",
        use_edge_weights=False,
        use_edge_attr=True,
        use_edge_attr_for_messages="positions+manhattan",
        edge_attr_processor="MLP",
        imitation_learning_model="MAGATPlus",
        cnn_mode="ResNetLarge_withMLP",
        module_residual=None,
        add_data_cost_to_go=True,
        add_data_greedy_action=False,
        add_data_num_previous_actions=None,
        normalize_cost_to_go=True,
        clamp_cost_to_go=1.0,
        clamped_values_doubled=False,
        train_on_terminated_agents=True,
        lr_start=MAGAT_LR,
        lr_end=MAGAT_LR_END,
        weight_decay=MAGAT_WEIGHT_DECAY,
    )


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        dilation=(1, 1),
        residual=True,
    ):
        super().__init__()
        self.conv1 = conv3x3(
            inplanes, planes, stride, padding=dilation[0], dilation=dilation[0]
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.residual = residual

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out = out + residual
        return self.relu(out)


class ResNet(torch.nn.Module):
    def __init__(
        self,
        layers,
        in_channels=4,
        num_classes=128,
        channels=(32, 64, 128, 128),
        pool_size=2,
    ):
        super().__init__()
        self.inplanes = channels[0]
        self.out_dim = channels[-1]

        self.conv1 = torch.nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(channels[0])
        self.relu = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=1)
        self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], stride=1)

        self.avgpool = torch.nn.AvgPool2d(pool_size)
        self.fc = torch.nn.Conv2d(
            self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )

    def _make_layer(
        self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                dilation=(
                    (1, 1)
                    if dilation == 1
                    else (dilation // 2 if new_level else dilation, dilation)
                ),
                residual=residual,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    residual=residual,
                    dilation=(dilation, dilation),
                )
            )
        return torch.nn.Sequential(*layers)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        for layer in self.layer1:
            layer.reset_parameters()
        for layer in self.layer2:
            layer.reset_parameters()
        for layer in self.layer3:
            layer.reset_parameters()
        self.fc.reset_parameters()

        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.fc(x)


class FixedResNetLargeWithMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet([1, 1, 1], in_channels=PYG_NUM_CHANNELS, num_classes=MAGAT_EMBEDDING_SIZE)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(1152, MAGAT_EMBEDDING_SIZE, bias=True)
        self.compress_mlp = torch.nn.ModuleList(
            [torch.nn.Linear(MAGAT_EMBEDDING_SIZE, MAGAT_EMBEDDING_SIZE, bias=True)]
        )

    def reset_parameters(self):
        self.resnet.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.compress_mlp:
            lin.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)
        for lin in self.compress_mlp:
            x = F.relu(lin(x))
        return x


class FixedEdgeAttrMLP(torch.nn.Module):
    """Match MAGAT-plus edge_attr_processor='MLP': 3 -> 32 -> 32 -> 128."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(MAGAT_EDGE_RAW_DIM, 32, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(32, MAGAT_EMBEDDING_SIZE, bias=True),
        )

    def reset_parameters(self):
        for module in self.net:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, edge_attr):
        return self.net(edge_attr)


class FixedMAGATPlusModel(torch.nn.Module):
    """Exact fixed-version MAGAT+ architecture needed by the external pipeline."""

    def __init__(self):
        super().__init__()
        self.cnn = FixedResNetLargeWithMLP()
        self.edge_attr_encoder = FixedEdgeAttrMLP()
        self.gnn = _get_gnn_module()(
            in_channels=MAGAT_EMBEDDING_SIZE,
            embedding_sizes=[MAGAT_EMBEDDING_SIZE] * MAGAT_NUM_GNN_LAYERS,
            num_attention_heads=MAGAT_NUM_ATTENTION_HEADS,
            num_gnn_layers=MAGAT_NUM_GNN_LAYERS,
            model_type="MAGATPlus",
            use_edge_weights=False,
            use_edge_attr=True,
            edge_dim=MAGAT_EMBEDDING_SIZE,
            model_residuals="all",
            attentionMode="MAGAT_multiplicative",
            use_edge_attr_for_messages="positions+manhattan",
        )
        self.actions_mlp = torch.nn.ModuleList(
            [
                torch.nn.Linear(MAGAT_EMBEDDING_SIZE, MAGAT_EMBEDDING_SIZE),
                torch.nn.Linear(MAGAT_EMBEDDING_SIZE, MAGAT_NUM_ACTIONS),
            ]
        )

    def reset_parameters(self):
        self.cnn.reset_parameters()
        self.edge_attr_encoder.reset_parameters()
        self.gnn.reset_parameters()
        for lin in self.actions_mlp:
            lin.reset_parameters()

    def forward(self, x, data):
        x = self.cnn(x)
        edge_index = data.edge_index
        edge_attr_raw = data.edge_attr
        if edge_index is None or edge_attr_raw is None:
            edge_index, edge_attr_raw = PyGBatchBuilder._materialize_valid_edges(
                data.edge_index_storage, data.edge_attr_storage, data.num_edges
            )
        edge_attr = None
        if edge_attr_raw is not None and edge_attr_raw.numel() > 0:
            edge_attr = self.edge_attr_encoder(edge_attr_raw)
        data.edge_index = edge_index
        data.edge_attr = edge_attr_raw
        x = self.gnn(x, data, edge_attr=edge_attr)
        x = F.relu(self.actions_mlp[0](x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.actions_mlp[1](x)


class MAGATRuntimeAdapter:
    """Fixed-version MAGAT+ model/loss/optimizer bundle for the external pipeline."""

    def __init__(
        self,
        device="cuda:0",
        lr: float = MAGAT_LR,
        lr_end: float = MAGAT_LR_END,
        lr_scheduler: str | None = None,
        scheduler_total_steps: int | None = None,
        grad_clip_norm: float | None = None,
        train_on_arrived_agents: bool = True,
    ):
        self.device = torch.device(device)
        self.args = build_fixed_magat_args()
        self.model = FixedMAGATPlusModel().to(self.device)
        self.model.reset_parameters()
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=MAGAT_WEIGHT_DECAY,
            capturable=True,
        )
        self.lr_start = float(lr)
        self.lr_end = float(lr_end)
        self.lr_scheduler_name = lr_scheduler or None
        self.scheduler_total_steps = (
            None if scheduler_total_steps is None else max(1, int(scheduler_total_steps))
        )
        self.grad_clip_norm = (
            None if grad_clip_norm is None or grad_clip_norm <= 0 else float(grad_clip_norm)
        )
        self.train_on_arrived_agents = bool(train_on_arrived_agents)
        self.scheduler = None
        if self.lr_scheduler_name is not None:
            if self.lr_scheduler_name != "cosine-annealing":
                raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler_name}")
            if self.scheduler_total_steps is None:
                raise ValueError(
                    "scheduler_total_steps is required when lr_scheduler is enabled"
                )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_total_steps,
                eta_min=self.lr_end,
            )
        self.batch_builder = PyGBatchBuilder()

    def _loss_from_logits(self, out: torch.Tensor, target: torch.Tensor, arrived: torch.Tensor) -> torch.Tensor:
        if not self.train_on_arrived_agents:
            keep_mask = ~arrived
            if not torch.any(keep_mask):
                return out.sum() * 0.0
            out = out[keep_mask]
            target = target[keep_mask]
        return self.loss_function(out, target)

    def _forward_loss_from_batch(self, data: RuntimePyGBatch) -> torch.Tensor:
        out = self.model(data.x, data)
        return self._loss_from_logits(out, data.y, data.arrived)

    def _forward_loss_from_stateful(self, simulator) -> torch.Tensor:
        data = self.batch_builder.build_from_stateful(simulator, materialize_edges=False)
        out = self.model(data.x, data)
        return self._loss_from_logits(out, data.y, data.arrived)

    def _step_optimizer(self, loss: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.detach()

    def build_batch(
        self,
        simulator,
        raw_batch: torch.Tensor,
        materialize_edges: bool = False,
    ) -> RuntimePyGBatch:
        return self.batch_builder.build(
            simulator,
            raw_batch,
            materialize_edges=materialize_edges,
        )

    def train_step_from_batch(self, data: RuntimePyGBatch) -> torch.Tensor:
        loss = self._forward_loss_from_batch(data)
        return self._step_optimizer(loss)

    def train_step(self, simulator, raw_batch: torch.Tensor) -> torch.Tensor:
        data = self.build_batch(simulator, raw_batch)
        return self.train_step_from_batch(data)

    def train_step_from_stateful(self, simulator) -> torch.Tensor:
        loss = self._forward_loss_from_stateful(simulator)
        return self._step_optimizer(loss)

    def loss_from_batch(self, data: RuntimePyGBatch) -> torch.Tensor:
        return self._forward_loss_from_batch(data).detach()

    def loss_from_stateful(self, simulator) -> torch.Tensor:
        return self._forward_loss_from_stateful(simulator).detach()

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def training_config(self) -> dict[str, float | str | int | bool | None]:
        return {
            "lr_start": self.lr_start,
            "lr_end": self.lr_end,
            "lr_scheduler": self.lr_scheduler_name,
            "scheduler_total_steps": self.scheduler_total_steps,
            "grad_clip_norm": self.grad_clip_norm,
            "train_on_arrived_agents": self.train_on_arrived_agents,
            "current_lr": self.current_lr(),
            "scheduler_last_epoch": None if self.scheduler is None else int(self.scheduler.last_epoch),
        }
