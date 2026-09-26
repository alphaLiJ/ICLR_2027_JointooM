"""Official-compatible MAPF-GPT model and online training adapter."""

from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from expert.mapf_gpt_schema import MAPFGPT_CONTEXT_SIZE, MAPFGPT_FEATURE_DIM


def _normalize_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


@dataclass
class GPTConfig:
    block_size: int = MAPFGPT_CONTEXT_SIZE
    vocab_size: int = 67
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False


def mapf_gpt_config(model_size: str) -> GPTConfig:
    normalized = str(model_size).upper()
    sizes = {
        "2M": (5, 5, 160),
        "6M": (8, 8, 256),
        "85M": (12, 12, 768),
    }
    if normalized not in sizes:
        raise ValueError(
            f"model_size must be one of {sorted(sizes)}, got {model_size!r}"
        )
    n_layer, n_head, n_embd = sizes[normalized]
    return GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd)


def default_mapf_gpt_microbatch_size(model_size: str, logical_batch_size: int) -> int:
    logical_batch_size = int(logical_batch_size)
    if logical_batch_size <= 0:
        raise ValueError(
            f"logical_batch_size must be positive, got {logical_batch_size}"
        )
    normalized = str(model_size).upper()
    defaults = {"2M": 256, "6M": 128, "85M": 16}
    if normalized not in defaults:
        raise ValueError(
            f"model_size must be one of {sorted(defaults)}, got {model_size!r}"
        )
    return min(logical_batch_size, defaults[normalized])


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            value, self.weight.shape, self.weight, self.bias, 1e-5
        )


class NonCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = value.size()
        query, key, val = self.c_attn(value).split(self.n_embd, dim=2)
        head_size = channels // self.n_head
        key = key.view(batch, sequence, self.n_head, head_size).transpose(1, 2)
        query = query.view(batch, sequence, self.n_head, head_size).transpose(1, 2)
        val = val.view(batch, sequence, self.n_head, head_size).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                val,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            weights = (query @ key.transpose(-2, -1)) * (
                1.0 / math.sqrt(key.size(-1))
            )
            weights = self.attn_dropout(F.softmax(weights, dim=-1))
            attended = weights @ val
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, channels
        )
        return self.resid_dropout(self.c_proj(attended))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(value))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(self.ln_1(value))
        return value + self.mlp(self.ln_2(value))


class GPT(nn.Module):
    """Parameter-compatible port of the official non-causal MAPF-GPT."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.transformer.wpe.weight.numel()
        return count

    def _hidden(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() != 2:
            raise ValueError(f"indices must have shape [B, T], got {indices.shape}")
        _, sequence = indices.shape
        if sequence > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(sequence, dtype=torch.long, device=indices.device)
        value = self.transformer.drop(
            self.transformer.wte(indices) + self.transformer.wpe(positions)
        )
        for block in self.transformer.h:
            value = block(value)
        return self.transformer.ln_f(value)

    def forward_all_logits(self, indices: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self._hidden(indices))

    def forward(
        self,
        indices: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self._hidden(indices)
        if targets is None:
            return self.lm_head(hidden[:, [-1], :]), None

        logits = self.lm_head(hidden)
        if targets.dim() == 1:
            valid_count = targets.ne(-1).sum()
            loss = F.cross_entropy(
                logits[:, -1, :],
                targets,
                ignore_index=-1,
                reduction="sum",
            ) / valid_count.clamp_min(1).to(logits.dtype)
        elif targets.dim() == 2:
            valid_count = targets.ne(-1).sum()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction="sum",
            ) / valid_count.clamp_min(1).to(logits.dtype)
        else:
            raise ValueError(
                f"targets must have shape [B] or [B,T], got {targets.shape}"
            )
        return logits, loss

    @torch.no_grad()
    def act(
        self,
        indices: torch.Tensor,
        *,
        do_sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        logits, _ = self(indices)
        action_logits = logits[:, -1, :5]
        if do_sample:
            probabilities = F.softmax(action_logits, dim=-1)
            return torch.multinomial(
                probabilities, num_samples=1, generator=generator
            ).squeeze(-1)
        return action_logits.argmax(dim=-1)

    def configure_optimizers(
        self,
        *,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        decay = [parameter for parameter in parameters.values() if parameter.dim() >= 2]
        no_decay = [parameter for parameter in parameters.values() if parameter.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        kwargs = {"fused": True} if fused_available and device_type == "cuda" else {}
        return torch.optim.AdamW(
            groups, lr=learning_rate, betas=betas, **kwargs
        )


class GPUMapfGPTShuffleBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        device: str | torch.device,
        seed: int,
    ):
        if int(capacity) <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.device = _normalize_device(device)
        self.tokens = torch.empty(
            (self.capacity, MAPFGPT_CONTEXT_SIZE),
            dtype=torch.int32,
            device=self.device,
        )
        self.labels = torch.empty(
            (self.capacity,), dtype=torch.int64, device=self.device
        )
        self.cursor = 0
        self.size = 0
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(int(seed))
        self._sample_order = torch.empty(
            self.capacity, dtype=torch.int64, device=self.device
        )
        self._sample_cursor = 0
        self._sample_domain_size = 0
        self.permutation_refreshes = 0

    def insert(self, tokens: torch.Tensor, labels: torch.Tensor) -> None:
        if tokens.dim() != 2 or tokens.shape[1] != MAPFGPT_CONTEXT_SIZE:
            raise ValueError(
                f"tokens must have shape [N, {MAPFGPT_CONTEXT_SIZE}], got {tokens.shape}"
            )
        if labels.shape != (tokens.shape[0],):
            raise ValueError(
                f"labels must have shape [{tokens.shape[0]}], got {labels.shape}"
            )
        if tokens.dtype != torch.int32 or labels.dtype != torch.int64:
            raise ValueError("tokens must be int32 and labels must be int64")
        if tokens.device != self.device or labels.device != self.device:
            raise ValueError("tokens and labels must be on the shuffle-buffer device")

        count = tokens.shape[0]
        previous_size = self.size
        if count >= self.capacity:
            self.tokens.copy_(tokens[-self.capacity :])
            self.labels.copy_(labels[-self.capacity :])
            self.cursor = 0
            self.size = self.capacity
            if previous_size != self.size:
                self._sample_cursor = 0
                self._sample_domain_size = 0
            return

        first = min(count, self.capacity - self.cursor)
        self.tokens[self.cursor : self.cursor + first].copy_(tokens[:first])
        self.labels[self.cursor : self.cursor + first].copy_(labels[:first])
        remaining = count - first
        if remaining:
            self.tokens[:remaining].copy_(tokens[first:])
            self.labels[:remaining].copy_(labels[first:])
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        if previous_size != self.size:
            self._sample_cursor = 0
            self._sample_domain_size = 0

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch_size)
        if batch_size <= 0 or batch_size > self.size:
            raise ValueError(
                f"batch_size must be in [1, {self.size}], got {batch_size}"
            )
        if (
            self._sample_domain_size != self.size
            or self._sample_cursor + batch_size > self.size
        ):
            self._sample_order[: self.size].copy_(
                torch.randperm(
                    self.size, device=self.device, generator=self.generator
                )
            )
            self._sample_cursor = 0
            self._sample_domain_size = self.size
            self.permutation_refreshes += 1
        indices = self._sample_order[
            self._sample_cursor : self._sample_cursor + batch_size
        ]
        self._sample_cursor += batch_size
        return self.tokens.index_select(0, indices), self.labels.index_select(0, indices)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "cursor": self.cursor,
            "size": self.size,
            "tokens": self.tokens[: self.size].detach().cpu().clone(),
            "labels": self.labels[: self.size].detach().cpu().clone(),
            "generator_state": self.generator.get_state().cpu(),
            "sample_order": self._sample_order[
                : self._sample_domain_size
            ].detach().cpu().clone(),
            "sample_cursor": self._sample_cursor,
            "sample_domain_size": self._sample_domain_size,
            "permutation_refreshes": self.permutation_refreshes,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        capacity = int(state["capacity"])
        if capacity != self.capacity:
            raise ValueError(
                "shuffle capacity mismatch: "
                f"checkpoint={capacity}, runtime={self.capacity}"
            )
        size = int(state["size"])
        cursor = int(state["cursor"])
        if not 0 <= size <= self.capacity:
            raise ValueError(f"invalid shuffle size {size}")
        if not 0 <= cursor < self.capacity:
            raise ValueError(f"invalid shuffle cursor {cursor}")
        tokens = torch.as_tensor(state["tokens"])
        labels = torch.as_tensor(state["labels"])
        if tokens.shape != (size, MAPFGPT_CONTEXT_SIZE):
            raise ValueError(
                "invalid checkpoint shuffle token shape: "
                f"expected {(size, MAPFGPT_CONTEXT_SIZE)}, got {tuple(tokens.shape)}"
            )
        if labels.shape != (size,):
            raise ValueError(
                "invalid checkpoint shuffle label shape: "
                f"expected {(size,)}, got {tuple(labels.shape)}"
            )
        if size:
            self.tokens[:size].copy_(tokens.to(self.device, dtype=torch.int32))
            self.labels[:size].copy_(labels.to(self.device, dtype=torch.int64))
        self.size = size
        self.cursor = cursor
        self.generator.set_state(torch.as_tensor(state["generator_state"]).cpu())
        sample_domain_size = int(state.get("sample_domain_size", 0))
        sample_cursor = int(state.get("sample_cursor", 0))
        if not 0 <= sample_domain_size <= self.size:
            raise ValueError(f"invalid shuffle sample domain {sample_domain_size}")
        if not 0 <= sample_cursor <= sample_domain_size:
            raise ValueError(f"invalid shuffle sample cursor {sample_cursor}")
        if sample_domain_size:
            sample_order = torch.as_tensor(state["sample_order"])
            if sample_order.shape != (sample_domain_size,):
                raise ValueError(
                    "invalid shuffle sample order shape: "
                    f"expected {(sample_domain_size,)}, got {tuple(sample_order.shape)}"
                )
            self._sample_order[:sample_domain_size].copy_(
                sample_order.to(self.device, dtype=torch.int64)
            )
        self._sample_domain_size = sample_domain_size
        self._sample_cursor = sample_cursor
        self.permutation_refreshes = int(state.get("permutation_refreshes", 0))


class MapfGPTRuntimeAdapter:
    def __init__(
        self,
        *,
        model_size: str = "2M",
        device: str | torch.device = "cuda",
        train_batch_size: int = 512,
        microbatch_size: int | None = None,
        shuffle_capacity: int = 8192,
        learning_rate: float = 6e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        grad_clip: float = 1.0,
        seed: int = 1337,
        train_on_arrived_agents: bool = True,
    ):
        self.model_size = str(model_size).upper()
        self.device = _normalize_device(device)
        self.train_batch_size = int(train_batch_size)
        if self.train_batch_size <= 0:
            raise ValueError(
                f"train_batch_size must be positive, got {train_batch_size}"
            )
        if microbatch_size is None:
            self.microbatch_size = default_mapf_gpt_microbatch_size(
                self.model_size, self.train_batch_size
            )
        else:
            self.microbatch_size = int(microbatch_size)
            if not 0 < self.microbatch_size <= self.train_batch_size:
                raise ValueError(
                    "microbatch_size must be in "
                    f"[1, {self.train_batch_size}], got {microbatch_size}"
                )
        self.grad_clip = float(grad_clip)
        self.train_on_arrived_agents = bool(train_on_arrived_agents)
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        self.config = mapf_gpt_config(self.model_size)
        self.model = GPT(self.config).to(self.device)
        self.optimizer = self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=self.device.type,
        )
        self.shuffle = GPUMapfGPTShuffleBuffer(
            capacity=shuffle_capacity,
            device=self.device,
            seed=seed,
        )
        self.optimizer_steps = 0
        self.samples_processed = 0
        self.supervised_samples_processed = 0
        self.stage_samples_seen = 0
        self.accepted_samples_seen = 0
        self.buffered_samples_seen = 0
        self.arrived_samples_seen = 0
        self.skipped_stages = 0
        self._masked_label_workspace: torch.Tensor | None = None
        self._stage_stats = torch.empty(5, dtype=torch.int64, device=self.device)

    def train_step(
        self, builder, raw_stage: torch.Tensor
    ) -> dict[str, float | int | bool | None]:
        builder.build_tokens(raw_stage)
        self._stage_stats[:4].copy_(builder.diagnostics)
        self._stage_stats[4] = builder.active_mask.sum(dtype=torch.int64)
        stage_stats = self._stage_stats.detach().cpu().tolist()
        diagnostics = stage_stats[:4]
        if any(diagnostics):
            raise RuntimeError(f"MAPF-GPT CUDA builder diagnostics are non-zero: {diagnostics}")
        stage_samples = int(builder.tokens.shape[0])
        active_samples = int(stage_stats[4])
        arrived_samples = stage_samples - active_samples
        tokens = builder.tokens
        if self.train_on_arrived_agents:
            labels = builder.labels
        else:
            if (
                self._masked_label_workspace is None
                or self._masked_label_workspace.shape != builder.labels.shape
                or self._masked_label_workspace.device != builder.labels.device
            ):
                self._masked_label_workspace = torch.empty_like(builder.labels)
            labels = self._masked_label_workspace
            labels.copy_(builder.labels)
            labels.masked_fill_(~builder.active_mask, -1)
        accepted_samples = int(tokens.shape[0])
        self.stage_samples_seen += stage_samples
        self.accepted_samples_seen += accepted_samples
        self.buffered_samples_seen += accepted_samples
        self.arrived_samples_seen += arrived_samples

        self.shuffle.insert(tokens, labels)
        batch_size = min(self.train_batch_size, self.shuffle.size)
        tokens, labels = self.shuffle.sample(batch_size)
        supervised_count = labels.ne(-1).sum()
        supervised_samples = int(supervised_count.item())

        if supervised_samples == 0:
            self.skipped_stages += 1
            return {
                "loss": None,
                "samples": 0,
                "stage_samples": stage_samples,
                "accepted_samples": accepted_samples,
                "arrived_samples": arrived_samples,
                "supervised_samples": 0,
                "microbatches": 0,
                "skipped": True,
                "optimizer_steps": self.optimizer_steps,
                "samples_processed": self.samples_processed,
            }

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        microbatches = 0
        for start in range(0, batch_size, self.microbatch_size):
            end = min(start + self.microbatch_size, batch_size)
            micro_labels = labels[start:end]
            micro_valid = micro_labels.ne(-1).sum()
            _, micro_loss = self.model(tokens[start:end], micro_labels)
            if micro_loss is None:
                raise RuntimeError("MAPF-GPT training forward returned no loss")
            scaled_loss = micro_loss * (
                micro_valid.to(micro_loss.dtype)
                / supervised_count.to(micro_loss.dtype)
            )
            scaled_loss.backward()
            accumulated_loss.add_(scaled_loss.detach().to(accumulated_loss.dtype))
            microbatches += 1
        loss_value = float(accumulated_loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite MAPF-GPT loss: {loss_value}")
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer_steps += 1
        self.samples_processed += batch_size
        self.supervised_samples_processed += supervised_samples
        return {
            "loss": loss_value,
            "samples": batch_size,
            "stage_samples": stage_samples,
            "accepted_samples": accepted_samples,
            "arrived_samples": arrived_samples,
            "supervised_samples": supervised_samples,
            "microbatches": microbatches,
            "skipped": False,
            "optimizer_steps": self.optimizer_steps,
            "samples_processed": self.samples_processed,
        }

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "model_size": self.model_size,
            "model_config": asdict(self.config),
            "train_batch_size": self.train_batch_size,
            "microbatch_size": self.microbatch_size,
            "raw_feature_dim": MAPFGPT_FEATURE_DIM,
            "history_length": 5,
            "cost_to_go_dtype": "uint16",
            "neighbor_order": "manhattan_then_agent_id",
            "train_on_arrived_agents": self.train_on_arrived_agents,
            "arrived_mask_source": "cuda_token_kernel",
            "arrived_mask_application": (
                "none" if self.train_on_arrived_agents else "loss_ignore_index"
            ),
            "ignore_index": -1,
            "optimizer_steps": self.optimizer_steps,
            "samples_processed": self.samples_processed,
            "supervised_samples_processed": self.supervised_samples_processed,
            "stage_samples_seen": self.stage_samples_seen,
            "accepted_samples_seen": self.accepted_samples_seen,
            "buffered_samples_seen": self.buffered_samples_seen,
            "arrived_samples_seen": self.arrived_samples_seen,
            "skipped_stages": self.skipped_stages,
        }

    def checkpoint_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "version": 1,
            "model_size": self.model_size,
            "train_batch_size": self.train_batch_size,
            "microbatch_size": self.microbatch_size,
            "train_on_arrived_agents": self.train_on_arrived_agents,
            "shuffle": self.shuffle.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "counters": {
                "optimizer_steps": self.optimizer_steps,
                "samples_processed": self.samples_processed,
                "supervised_samples_processed": self.supervised_samples_processed,
                "stage_samples_seen": self.stage_samples_seen,
                "accepted_samples_seen": self.accepted_samples_seen,
                "buffered_samples_seen": self.buffered_samples_seen,
                "arrived_samples_seen": self.arrived_samples_seen,
                "skipped_stages": self.skipped_stages,
            },
        }
        if self.device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state(self.device).cpu()
        return state

    def load_checkpoint_state(self, state: dict[str, object]) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError(
                f"unsupported MAPF-GPT runtime checkpoint version {state.get('version')}"
            )
        expected = {
            "model_size": self.model_size,
            "train_batch_size": self.train_batch_size,
            "microbatch_size": self.microbatch_size,
            "train_on_arrived_agents": self.train_on_arrived_agents,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"runtime checkpoint {key} mismatch: "
                    f"checkpoint={state.get(key)!r}, runtime={value!r}"
                )
        self.shuffle.load_state_dict(state["shuffle"])
        counters = dict(state["counters"])
        for name in (
            "optimizer_steps",
            "samples_processed",
            "supervised_samples_processed",
            "stage_samples_seen",
            "accepted_samples_seen",
            "arrived_samples_seen",
            "skipped_stages",
        ):
            setattr(self, name, int(counters[name]))
        self.buffered_samples_seen = int(
            counters.get("buffered_samples_seen", counters["accepted_samples_seen"])
        )
        torch.set_rng_state(torch.as_tensor(state["torch_rng_state"]).cpu())
        if self.device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(
                torch.as_tensor(state["cuda_rng_state"]).cpu(), self.device
            )

    def load_checkpoint(self, path: str) -> dict[str, object]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"invalid MAPF-GPT checkpoint payload at {path!r}")
        self.model.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        runtime_state = payload.get("runtime_state")
        if runtime_state is not None:
            self.load_checkpoint_state(runtime_state)
        elif payload.get("optimizer_step") is not None:
            self.optimizer_steps = int(payload["optimizer_step"])
        return payload


__all__ = [
    "GPT",
    "GPTConfig",
    "GPUMapfGPTShuffleBuffer",
    "MapfGPTRuntimeAdapter",
    "NonCausalSelfAttention",
    "default_mapf_gpt_microbatch_size",
    "mapf_gpt_config",
]
