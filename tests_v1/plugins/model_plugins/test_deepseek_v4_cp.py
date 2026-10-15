# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from llamafactory.v1.plugins.model_plugins.parallelization import deepseek_v4 as sequence_parallel
from llamafactory.v1.plugins.model_plugins.parallelization.deepseek_v4 import _get_deepseek_v4_eager_attention_fn


def test_deepseek_v4_eager_attention_adapts_ulysses_heads_and_mask(monkeypatch: pytest.MonkeyPatch):
    group = object()
    module = SimpleNamespace(
        config=SimpleNamespace(sliding_window=2),
        num_key_value_groups=4,
        sinks=torch.arange(4, dtype=torch.float32),
        sliding_window=2,
        training=True,
    )
    query = torch.zeros(1, 2, 4, 3)
    local_mask = torch.tensor([[[[0.0, torch.finfo(torch.float32).min], [0.0, 0.0]]]])
    captured = {}

    monkeypatch.setattr(sequence_parallel.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(sequence_parallel.dist, "get_rank", lambda _: 1)

    def fake_all_gather(outputs, tensor, **_):
        for output in outputs:
            output.copy_(tensor)

    monkeypatch.setattr(sequence_parallel.dist, "all_gather", fake_all_gather)

    def eager_attention(module, query, key, value, attention_mask, **kwargs):
        captured["num_key_value_groups"] = module.num_key_value_groups
        captured["sinks"] = module.sinks
        captured["attention_mask"] = attention_mask
        return query, None

    wrapped = _get_deepseek_v4_eager_attention_fn(eager_attention, group)
    output, _ = wrapped(module, query, query, query, local_mask)

    assert output is query
    assert captured["num_key_value_groups"] == 1
    assert captured["sinks"].tolist() == [2.0, 3.0]
    assert captured["attention_mask"].shape == (1, 1, 4, 4)
    assert (captured["attention_mask"][0, 0] == 0).tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, True, True, False],
        [False, False, True, True],
    ]


def test_deepseek_v4_cp_plan_matches_backend_api(monkeypatch):
    pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
    backend = pytest.importorskip("fsdp_turbo.distributed.context_parallel.ulysses.ulysses_context_parallel")
    from fsdp_turbo.utils import patch

    captured = {}
    monkeypatch.setattr(sequence_parallel, "_prepare_deepseek_v4_compressors", lambda *args: None)
    monkeypatch.setattr(patch, "patch_namespace_members", lambda *args: None)
    monkeypatch.setattr(backend.torch.distributed, "get_world_size", lambda group: 2)

    def replace(target, group, wrap_fn, wrap_kwargs):
        captured.update(target=target, wrap_fn=wrap_fn, wrap_kwargs=wrap_kwargs)

    monkeypatch.setattr(backend, "_import_and_replace_fn", replace)
    sequence_parallel.apply_deepseek_v4_sequence_parallel(object(), object())
    assert captured["target"].endswith("modeling_deepseek_v4.eager_attention_forward")
    assert captured["wrap_fn"] is backend.get_ulysses_compressor_attention_fn
    assert captured["wrap_kwargs"] == {"attn_output_transposed": True}


def test_deepseek_v4_adapter_matches_native_eager_and_sink_gradients(monkeypatch):
    eager_attention_forward = pytest.importorskip(
        "transformers.models.deepseek_v4.modeling_deepseek_v4"
    ).eager_attention_forward

    monkeypatch.setattr(sequence_parallel.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(sequence_parallel.dist, "get_rank", lambda _: 1)
    sinks = torch.arange(4, dtype=torch.float32, requires_grad=True)
    module = SimpleNamespace(sinks=sinks, num_key_value_groups=4, training=False)
    query = torch.randn(1, 2, 4, 3, requires_grad=True)
    key = torch.randn(1, 2, 4, 3, requires_grad=True)
    mask = torch.zeros(1, 1, 4, 4)
    expected_module = SimpleNamespace(sinks=sinks[2:], num_key_value_groups=1, training=False)
    expected, _ = eager_attention_forward(expected_module, query, key, key, mask, scaling=0.5)
    actual, _ = _get_deepseek_v4_eager_attention_fn(eager_attention_forward, object())(
        module, query, key, key, mask, scaling=0.5
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.equal(sinks.grad[:2], torch.zeros(2))
    assert sinks.grad[2:].abs().sum() > 0
    assert module.num_key_value_groups == 4


def test_deepseek_v4_uses_shared_language_split_hook(monkeypatch):
    from llamafactory.v1.plugins.model_plugins.parallelization import hook
    from llamafactory.v1.plugins.model_plugins.parallelization import sequence_parallel as routing

    class Mesh:
        def __getitem__(self, name):
            return self

        def size(self):
            return 2

        def get_local_rank(self):
            return 1

    class Core(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="deepseek_v4", _attn_implementation="eager")
            self.base_model = Core()

    interface = SimpleNamespace(get_device_mesh=lambda _: Mesh(), get_group=lambda _: object())
    monkeypatch.setattr(hook, "DistributedInterface", lambda: interface)
    monkeypatch.setattr(routing, "DistributedInterface", lambda: interface)
    calls = []
    monkeypatch.setattr(sequence_parallel, "apply_deepseek_v4_sequence_parallel", lambda *args: calls.append(args))
    model = Model()
    routing.apply_sequence_parallel(model, 2)
    batch = model.base_model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.ones(1, 4),
        position_ids=torch.arange(4).reshape(1, 4),
    )
    assert batch["input_ids"].tolist() == [[3, 4]]
    assert batch["position_ids"].tolist() == [[2, 3]]
    assert batch["attention_mask"].shape == (1, 2)
    assert batch["use_cache"] is False
    assert len(calls) == 1
    assert len(model.base_model._forward_pre_hooks) == 1
