from __future__ import annotations

import json

import torch


def test_infer_actions_preserves_order_across_microbatches():
    from expert.minimal_p0_mapf_gpt_resident_runner import _infer_actions

    class Model:
        def __init__(self):
            self.batch_sizes = []

        def act(self, tokens, *, do_sample):
            assert do_sample is False
            self.batch_sizes.append(int(tokens.shape[0]))
            return tokens[:, 0].remainder(5).to(torch.int64)

    model = Model()
    tokens = torch.arange(11, dtype=torch.int32)[:, None].repeat(1, 256)
    output = torch.empty(11, dtype=torch.uint8)

    count = _infer_actions(model, tokens, output, microbatch_size=4)

    assert count == 3
    assert model.batch_sizes == [4, 4, 3]
    assert output.tolist() == [value % 5 for value in range(11)]


def test_failed_report_can_be_written_without_performance_fields(tmp_path):
    from expert.minimal_p0_mapf_gpt_resident_runner import _write

    output = tmp_path / "failed"
    _write(
        output,
        {
            "schema_version": 1,
            "status": "failed",
            "exception_type": "RuntimeError",
            "exception_message": "diagnostic",
        },
    )

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "stdout.log").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert summary == {
        "exception_message": "diagnostic",
        "exception_type": "RuntimeError",
        "status": "failed",
    }


def test_official_checkpoint_prefix_and_model_args_are_supported(tmp_path):
    from expert.mapf_gpt_runtime import GPT, mapf_gpt_config
    from expert.minimal_p0_mapf_gpt_resident_runner import _load_inference_model

    config = mapf_gpt_config("2M")
    source = GPT(config)
    checkpoint = tmp_path / "MAPF-GPT-2M.pt"
    torch.save(
        {
            "model": {
                f"_orig_mod.{key}": value for key, value in source.state_dict().items()
            },
            "model_args": vars(config),
            "iter_num": 123,
        },
        checkpoint,
    )

    loaded, payload, model_size = _load_inference_model(
        checkpoint, device="cpu"
    )

    assert payload["iter_num"] == 123
    assert model_size == "MAPF-GPT-2M"
    for key, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
