from types import SimpleNamespace

import numpy as np

from faceswap.swapping import _run_swap_model


class _Session:
    def __init__(self, provider):
        self.provider = provider
        self.calls = 0

    def get_providers(self):
        return [self.provider]

    def run(self, outputs, inputs):
        self.calls += 1
        return [np.ones((1, 3, 2, 2), np.float32)]


def test_swap_model_retains_standard_cpu_path():
    session = _Session("CPUExecutionProvider")
    model = SimpleNamespace(
        session=session,
        input_names=["image", "latent"],
        output_names=["output"],
    )

    output = _run_swap_model(
        model,
        np.zeros((1, 3, 2, 2), np.float32),
        np.zeros((1, 2), np.float32),
    )

    assert output.shape == (1, 3, 2, 2)
    assert session.calls == 1
