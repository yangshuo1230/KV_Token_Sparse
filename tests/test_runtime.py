from types import SimpleNamespace

import torch

from kvstudy.runtime import DecodeEngine


class _Backend:
    name = "test"

    def __init__(self):
        self.installs = 0
        self.steps = []

    def install(self, model):
        self.installs += 1

    def before_step(self, model, step):
        self.steps.append(step)


class _Model:
    def __call__(self, input_ids, position_ids, past_key_values, **kwargs):
        value = float(input_ids.item())
        logits = torch.tensor([[[value, float(position_ids.item())]]])
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_decode_engine_owns_the_shared_autoregressive_loop():
    backend = _Backend()
    logits = DecodeEngine(_Model()).decode(
        cache=object(),
        query_ids=torch.tensor([[3, 4, 5]]),
        start_position=10,
        backend=backend,
    )
    assert backend.installs == 1
    assert backend.steps == [0, 1, 2]
    assert logits.tolist() == [[3.0, 10.0], [4.0, 11.0], [5.0, 12.0]]
