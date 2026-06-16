import pytest

from gaze_query_rag.modeling.devices import get_torch_device_status, resolve_torch_device


def test_resolve_torch_device_accepts_cpu() -> None:
    assert str(resolve_torch_device("cpu")) == "cpu"


def test_resolve_torch_device_auto_returns_available_backend() -> None:
    assert str(resolve_torch_device("auto")) in {"cpu", "cuda", "mps"}


def test_explicit_mps_fails_clearly_when_unusable() -> None:
    status = get_torch_device_status()
    if status.mps_usable:
        pytest.skip("MPS is usable in this runtime.")

    with pytest.raises(RuntimeError, match="MPS was requested"):
        resolve_torch_device("mps")
