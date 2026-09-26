import importlib


def test_fixed_magat_plus_runtime_imports():
    mod = importlib.import_module("expert.fixed_magat_plus_runtime")
    assert hasattr(mod, "PyGBatchBuilder")
    assert hasattr(mod, "MAGATRuntimeAdapter")


def test_expert_running_imports():
    mod = importlib.import_module("expert.expert_running")
    assert hasattr(mod, "GPUComputeHandler")
    assert hasattr(mod, "ExtremeMAPFPipeline")


def test_topology_training_module_imports():
    mod = importlib.import_module("mapf_cuda.training.topology_async")
    assert hasattr(mod, "benchmark_topology_async_training_system")
    assert hasattr(mod, "_partition_expert_assignments")
