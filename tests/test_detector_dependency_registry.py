from pathlib import Path

from selective_detection.detector_dependency_registry import default_adapters


def test_default_detector_adapters_reference_official_paths():
    adapters = {adapter.name: adapter for adapter in default_adapters(Path("."))}

    assert adapters["univfd"].checkpoint_path == Path(
        "third_party/UniversalFakeDetect/pretrained_weights/fc_weights.pth"
    )
    assert adapters["safe"].checkpoint_path == Path("third_party/SAFE/checkpoint/checkpoint-best.pth")
    assert adapters["univfd"].entrypoint.name == "validate.py"
    assert adapters["safe"].entrypoint.name == "main_finetune.py"
