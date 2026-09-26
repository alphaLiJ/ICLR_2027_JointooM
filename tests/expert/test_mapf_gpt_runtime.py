import pytest
import torch


class _StaticTokenBuilder:
    def __init__(self, tokens, labels, active_mask=None):
        self.tokens = tokens
        self.labels = labels
        self.active_mask = (
            torch.ones_like(labels, dtype=torch.bool)
            if active_mask is None
            else active_mask
        )
        self.diagnostics = torch.zeros(4, dtype=torch.int32, device=tokens.device)

    def build_tokens(self, raw_stage):
        return None


def test_official_model_size_configs_and_absolute_position_embedding():
    from expert.mapf_gpt_runtime import GPT, mapf_gpt_config

    expected = {
        "2M": (5, 5, 160),
        "6M": (8, 8, 256),
        "85M": (12, 12, 768),
    }
    for size, values in expected.items():
        config = mapf_gpt_config(size)
        assert (config.n_layer, config.n_head, config.n_embd) == values
        assert config.block_size == 256
        assert config.vocab_size == 67
        assert config.dropout == 0.0
        assert config.bias is False

    model = GPT(mapf_gpt_config("2M"))
    assert isinstance(model.transformer.wpe, torch.nn.Embedding)
    assert model.transformer.wpe.weight.shape == (256, 160)
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_attention_is_noncausal_and_future_token_changes_first_position():
    from expert.mapf_gpt_runtime import GPT, GPTConfig

    torch.manual_seed(7)
    model = GPT(GPTConfig(block_size=4, n_layer=1, n_head=1, n_embd=8))
    model.eval()
    left = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    changed_future = torch.tensor([[1, 2, 3, 5]], dtype=torch.int32)

    all_left = model.forward_all_logits(left)
    all_changed = model.forward_all_logits(changed_future)
    assert not torch.equal(all_left[:, 0], all_changed[:, 0])


def test_training_loss_uses_last_position_and_full_67_way_vocabulary():
    from expert.mapf_gpt_runtime import GPT, GPTConfig

    model = GPT(GPTConfig(block_size=8, n_layer=1, n_head=1, n_embd=8))
    tokens = torch.randint(0, 67, (3, 8), dtype=torch.int32)
    labels = torch.tensor([0, 2, 4], dtype=torch.int64)
    logits, loss = model(tokens, labels)

    assert logits.shape == (3, 8, 67)
    expected = torch.nn.functional.cross_entropy(logits[:, -1, :], labels)
    assert torch.allclose(loss, expected)


def test_one_dimensional_targets_ignore_arrived_label_minus_one():
    from expert.mapf_gpt_runtime import GPT, GPTConfig

    torch.manual_seed(17)
    model = GPT(GPTConfig(block_size=8, n_layer=1, n_head=1, n_embd=8))
    tokens = torch.randint(0, 67, (3, 8), dtype=torch.int32)
    labels = torch.tensor([0, -1, 4], dtype=torch.int64)

    logits, loss = model(tokens, labels)

    expected = torch.nn.functional.cross_entropy(
        logits[[0, 2], -1, :], labels[[0, 2]]
    )
    assert torch.allclose(loss, expected)


def test_two_dimensional_all_ignored_targets_return_finite_zero_loss():
    from expert.mapf_gpt_runtime import GPT, GPTConfig

    model = GPT(GPTConfig(block_size=8, n_layer=1, n_head=1, n_embd=8))
    tokens = torch.randint(0, 67, (2, 8), dtype=torch.int32)
    labels = torch.full((2, 8), -1, dtype=torch.int64)

    _, loss = model(tokens, labels)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_gpu_shuffle_buffer_preserves_token_label_pairs_across_wraparound():
    from expert.mapf_gpt_runtime import GPUMapfGPTShuffleBuffer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = GPUMapfGPTShuffleBuffer(capacity=5, device=device, seed=13)
    tokens = torch.arange(8, device=device, dtype=torch.int32)[:, None].repeat(1, 256)
    labels = torch.arange(8, device=device, dtype=torch.int64)
    buffer.insert(tokens[:3], labels[:3])
    buffer.insert(tokens[3:], labels[3:])

    assert buffer.size == 5
    sampled_tokens, sampled_labels = buffer.sample(5)
    assert sampled_tokens.shape == (5, 256)
    assert sampled_labels.shape == (5,)
    assert torch.equal(sampled_tokens[:, 0].to(torch.int64), sampled_labels)
    assert set(sampled_labels.cpu().tolist()) == {3, 4, 5, 6, 7}


def test_gpu_shuffle_buffer_amortizes_permutation_generation():
    from expert.mapf_gpt_runtime import GPUMapfGPTShuffleBuffer

    buffer = GPUMapfGPTShuffleBuffer(capacity=8, device="cpu", seed=17)
    tokens = torch.arange(8, dtype=torch.int32)[:, None].repeat(1, 256)
    labels = torch.arange(8, dtype=torch.int64)
    buffer.insert(tokens, labels)

    first_tokens, first_labels = buffer.sample(4)
    second_tokens, second_labels = buffer.sample(4)

    assert buffer.permutation_refreshes == 1
    sampled_labels = torch.cat((first_labels, second_labels))
    sampled_tokens = torch.cat((first_tokens[:, 0], second_tokens[:, 0])).to(torch.int64)
    assert torch.equal(sampled_tokens, sampled_labels)
    assert set(sampled_labels.tolist()) == set(range(8))

    buffer.sample(4)
    assert buffer.permutation_refreshes == 2


def test_model_specific_microbatch_defaults_are_bounded_by_logical_batch():
    from expert.mapf_gpt_runtime import default_mapf_gpt_microbatch_size

    assert default_mapf_gpt_microbatch_size("2M", 1024) == 256
    assert default_mapf_gpt_microbatch_size("6M", 1024) == 128
    assert default_mapf_gpt_microbatch_size("85M", 1024) == 16
    assert default_mapf_gpt_microbatch_size("2M", 32) == 32


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"train_batch_size": 0}, "train_batch_size"),
        ({"train_batch_size": 4, "microbatch_size": 0}, "microbatch_size"),
        ({"train_batch_size": 4, "microbatch_size": 5}, "microbatch_size"),
    ],
)
def test_runtime_rejects_invalid_logical_and_microbatch_sizes(kwargs, message):
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    with pytest.raises(ValueError, match=message):
        MapfGPTRuntimeAdapter(model_size="2M", device="cpu", **kwargs)


def test_microbatch_accumulation_matches_full_logical_batch_update():
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    generator = torch.Generator().manual_seed(23)
    tokens = torch.randint(0, 67, (4, 256), dtype=torch.int32, generator=generator)
    labels = torch.tensor([0, 1, -1, 4], dtype=torch.int64)
    full = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cpu",
        train_batch_size=4,
        microbatch_size=4,
        shuffle_capacity=4,
        learning_rate=1e-4,
        seed=31,
        train_on_arrived_agents=True,
    )
    split = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cpu",
        train_batch_size=4,
        microbatch_size=2,
        shuffle_capacity=4,
        learning_rate=1e-4,
        seed=31,
        train_on_arrived_agents=True,
    )

    full_metrics = full.train_step(_StaticTokenBuilder(tokens, labels), tokens)
    split_metrics = split.train_step(_StaticTokenBuilder(tokens, labels), tokens)

    assert full_metrics["optimizer_steps"] == split_metrics["optimizer_steps"] == 1
    assert full_metrics["microbatches"] == 1
    assert split_metrics["microbatches"] == 2
    assert full_metrics["supervised_samples"] == split_metrics["supervised_samples"] == 3
    assert full_metrics["loss"] == pytest.approx(split_metrics["loss"], rel=1e-5)
    for full_parameter, split_parameter in zip(
        full.model.parameters(), split.model.parameters()
    ):
        assert torch.allclose(full_parameter, split_parameter, atol=1e-5, rtol=1e-5)


def test_runtime_checkpoint_state_restores_shuffle_rng_and_counters():
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    tokens = torch.arange(6, dtype=torch.int32)[:, None].repeat(1, 256) % 67
    labels = torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.int64)
    runtime = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cpu",
        train_batch_size=4,
        microbatch_size=2,
        shuffle_capacity=6,
        learning_rate=1e-4,
        seed=41,
    )
    runtime.train_step(_StaticTokenBuilder(tokens, labels), tokens)
    state = runtime.checkpoint_state()

    restored = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cpu",
        train_batch_size=4,
        microbatch_size=2,
        shuffle_capacity=6,
        learning_rate=1e-4,
        seed=999,
    )
    restored.load_checkpoint_state(state)

    assert restored.optimizer_steps == runtime.optimizer_steps
    assert restored.samples_processed == runtime.samples_processed
    assert restored.supervised_samples_processed == runtime.supervised_samples_processed
    assert restored.shuffle.cursor == runtime.shuffle.cursor
    assert restored.shuffle.size == runtime.shuffle.size
    assert torch.equal(restored.shuffle.tokens, runtime.shuffle.tokens)
    assert torch.equal(restored.shuffle.labels, runtime.shuffle.labels)
    expected_tokens, expected_labels = runtime.shuffle.sample(4)
    actual_tokens, actual_labels = restored.shuffle.sample(4)
    assert torch.equal(actual_tokens, expected_tokens)
    assert torch.equal(actual_labels, expected_labels)


def test_arrived_label_mask_reuses_fixed_workspace():
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    tokens = torch.arange(4, dtype=torch.int32)[:, None].repeat(1, 256) % 67
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    active = torch.tensor([True, False, True, False])
    builder = _StaticTokenBuilder(tokens, labels, active)
    runtime = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cpu",
        train_batch_size=4,
        microbatch_size=2,
        shuffle_capacity=8,
        learning_rate=1e-4,
        seed=47,
        train_on_arrived_agents=False,
    )

    runtime.train_step(builder, tokens)
    first_pointer = runtime._masked_label_workspace.data_ptr()
    runtime.train_step(builder, tokens)

    assert runtime._masked_label_workspace.data_ptr() == first_pointer


def test_runtime_load_checkpoint_restores_model_optimizer_and_runtime_state(tmp_path):
    from expert.checkpoint_manager import TopKCheckpointManager
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    tokens = torch.arange(4, dtype=torch.int32)[:, None].repeat(1, 256) % 67
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    runtime = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cpu",
        train_batch_size=4,
        microbatch_size=2,
        shuffle_capacity=4,
        learning_rate=1e-4,
        seed=43,
    )
    metrics = runtime.train_step(_StaticTokenBuilder(tokens, labels), tokens)
    manager = TopKCheckpointManager(tmp_path, save_interval_steps=1)
    saved = manager.maybe_save(
        loss=metrics["loss"], runtime=runtime, optimizer_step=runtime.optimizer_steps
    )

    restored = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cpu",
        train_batch_size=4,
        microbatch_size=2,
        shuffle_capacity=4,
        learning_rate=1e-4,
        seed=999,
    )
    payload = restored.load_checkpoint(saved.path)

    assert payload["optimizer_step"] == 1
    assert restored.optimizer_steps == 1
    assert restored.shuffle.size == 4
    for expected, actual in zip(runtime.model.parameters(), restored.model.parameters()):
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_runtime_train_step_updates_model_and_reports_schema_metadata():
    import grid_world_cpp as ext
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    grids = torch.zeros(
        (1, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda",
    )
    builder = ext.MapfGPTObservationBuilder(grids, 4)
    rows = torch.zeros((4, 13), dtype=torch.int16, device="cuda")
    rows[:, 1] = torch.arange(4, dtype=torch.int16, device="cuda")
    rows[:, 2] = 10
    rows[:, 3] = torch.arange(4, dtype=torch.int16, device="cuda") + 10
    rows[:, 4] = 20
    rows[:, 5] = torch.arange(4, dtype=torch.int16, device="cuda") + 10
    rows[:, 6] = torch.arange(4, dtype=torch.int16, device="cuda") % 5
    rows[:, 7] = 1
    rows[:, 8:13] = 5
    runtime = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cuda",
        train_batch_size=4,
        shuffle_capacity=8,
        learning_rate=1e-4,
        seed=5,
    )
    before = runtime.model.transformer.wte.weight.detach().clone()

    metrics = runtime.train_step(builder, rows)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["samples"] == 4
    assert metrics["stage_samples"] == 4
    assert metrics["accepted_samples"] == 4
    assert metrics["supervised_samples"] == 4
    assert metrics["skipped"] is False
    assert not torch.equal(before, runtime.model.transformer.wte.weight)
    metadata = runtime.checkpoint_metadata()
    assert metadata["raw_feature_dim"] == 13
    assert metadata["cost_to_go_dtype"] == "uint16"
    assert metadata["neighbor_order"] == "manhattan_then_agent_id"
    assert metadata["model_size"] == "2M"
    assert metadata["train_on_arrived_agents"] is True
    assert metadata["arrived_mask_source"] == "cuda_token_kernel"
    assert metadata["arrived_mask_application"] == "none"
    assert metadata["ignore_index"] == -1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_runtime_optionally_masks_arrived_loss_without_compacting_network_batch():
    import grid_world_cpp as ext
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    grids = torch.zeros(
        (1, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda",
    )
    builder = ext.MapfGPTObservationBuilder(grids, 4)
    rows = torch.zeros((4, 13), dtype=torch.int16, device="cuda")
    rows[:, 1] = torch.arange(4, dtype=torch.int16, device="cuda")
    rows[:, 2] = 10
    rows[:, 3] = torch.arange(4, dtype=torch.int16, device="cuda") + 10
    rows[:, 4] = 20
    rows[:, 5] = rows[:, 3]
    rows[:, 6] = torch.arange(4, dtype=torch.int16, device="cuda")
    rows[:, 7] = 1
    rows[:, 8:13] = 5
    rows[0, 4:6] = rows[0, 2:4]
    rows[2, 4:6] = rows[2, 2:4]
    runtime = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cuda",
        train_batch_size=4,
        shuffle_capacity=8,
        learning_rate=1e-4,
        seed=11,
        train_on_arrived_agents=False,
    )

    metrics = runtime.train_step(builder, rows)
    assert metrics["stage_samples"] == 4
    assert metrics["accepted_samples"] == 4
    assert metrics["arrived_samples"] == 2
    assert metrics["samples"] == 4
    assert metrics["supervised_samples"] == 2
    assert metrics["skipped"] is False
    assert runtime.shuffle.size == 4
    assert runtime.shuffle.labels[:4].cpu().tolist() == [-1, 1, -1, 3]
    assert runtime.samples_processed == 4
    assert runtime.supervised_samples_processed == 2
    assert runtime.checkpoint_metadata()["arrived_mask_application"] == "loss_ignore_index"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_runtime_buffers_all_arrived_stage_but_skips_all_ignored_batch():
    import grid_world_cpp as ext
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    grids = torch.zeros(
        (1, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda",
    )
    builder = ext.MapfGPTObservationBuilder(grids, 4)
    all_arrived = torch.zeros((4, 13), dtype=torch.int16, device="cuda")
    all_arrived[:, 1] = torch.arange(4, dtype=torch.int16, device="cuda")
    all_arrived[:, 2] = 10
    all_arrived[:, 3] = torch.arange(4, dtype=torch.int16, device="cuda") + 10
    all_arrived[:, 4:6] = all_arrived[:, 2:4]
    all_arrived[:, 6] = torch.arange(4, dtype=torch.int16, device="cuda")
    all_arrived[:, 7] = 1
    all_arrived[:, 8:13] = 5
    runtime = MapfGPTRuntimeAdapter(
        model_size="2M",
        device="cuda",
        train_batch_size=4,
        shuffle_capacity=4,
        learning_rate=1e-4,
        seed=19,
        train_on_arrived_agents=False,
    )

    skipped = runtime.train_step(builder, all_arrived)
    assert skipped["stage_samples"] == 4
    assert skipped["accepted_samples"] == 4
    assert skipped["arrived_samples"] == 4
    assert skipped["samples"] == 0
    assert skipped["supervised_samples"] == 0
    assert skipped["skipped"] is True
    assert skipped["loss"] is None
    assert runtime.optimizer_steps == 0
    assert runtime.skipped_stages == 1
    assert runtime.shuffle.size == 4
    assert runtime.shuffle.labels[:4].cpu().tolist() == [-1, -1, -1, -1]
    assert runtime.samples_processed == 0
