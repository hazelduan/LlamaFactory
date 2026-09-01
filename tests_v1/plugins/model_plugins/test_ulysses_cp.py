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
import torch.multiprocessing as mp

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.config.training_args import TrainingArguments
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.plugins.model_plugins.parallelization import sequence_parallel, ulysses
from llamafactory.v1.plugins.model_plugins.parallelization.sequence_parallel import (
    SequenceParallelModelPlugin,
    _get_deepseek_v4_eager_attention_fn,
    sequence_parallel_loss,
)
from llamafactory.v1.utils.env import find_available_port
from llamafactory.v1.utils.pytest import dist_env


def test_qwen3_5_broadcast_position_ids_keep_packed_boundaries(monkeypatch: pytest.MonkeyPatch):
    local_position_ids = torch.tensor([[0, 1, 0]])
    remote_position_ids = torch.tensor([[1, 2, 3]])
    mrope_position_ids = local_position_ids.unsqueeze(0).expand(3, -1, -1)
    captured = {}

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", lambda _, tensor, *__: tensor)
    monkeypatch.setattr(ulysses, "get_ulysses_sequence_parallel_world_size", lambda _: 2)

    def fake_all_gather(outputs, tensor, **_):
        outputs[0].copy_(tensor)
        outputs[1].copy_(remote_position_ids if tensor.shape == local_position_ids.shape else tensor)

    def fake_attention(query, _key, _value, _attention_mask, **kwargs):
        captured["position_ids"] = kwargs["position_ids"]
        return query

    monkeypatch.setattr(ulysses.dist, "all_gather", fake_all_gather)
    attention = ulysses.UlyssesAttention(sequence_process_group=object(), attn_fn=fake_attention)
    hidden_states = torch.zeros(1, 3, 2, 4)

    attention(hidden_states, hidden_states, hidden_states, None, 6, position_ids=mrope_position_ids)

    assert captured["position_ids"].tolist() == [[0, 1, 0, 1, 2, 3]]
    assert captured["position_ids"].is_contiguous()


def test_true_mrope_position_ids_are_not_used_as_packed_boundaries():
    mrope_position_ids = torch.tensor([[[0, 1, 2]], [[0, 1, 1]], [[0, 1, 0]]])

    assert ulysses._get_text_position_ids(mrope_position_ids) is None


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
        captured["s_aux"] = kwargs["s_aux"]
        return query, None

    wrapped = _get_deepseek_v4_eager_attention_fn(eager_attention, group)
    output, _ = wrapped(module, query, query, query, local_mask)

    assert output is query
    assert captured["num_key_value_groups"] == 1
    assert captured["sinks"].tolist() == [2.0, 3.0]
    assert torch.equal(captured["sinks"], captured["s_aux"])
    assert captured["attention_mask"].shape == (1, 1, 4, 4)
    assert (captured["attention_mask"][0, 0] == 0).tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, True, True, False],
        [False, False, True, True],
    ]


def _test_sequence_parallel_loss(
    local_rank: int, world_size: int, master_port: int, cp_size: int, dp_size: int, batch_size: int
):
    with dist_env(local_rank, world_size, master_port):
        model_args = ModelArguments(model="llamafactory/tiny-random-qwen3")

        training_args = TrainingArguments(cp_mode="ulysses", cp_size=cp_size, dp_size=dp_size)
        DistributedInterface(training_args)

        # Now create model engine
        model_engine = ModelEngine(model_args=model_args)

        # Apply sequence parallel plugin
        SequenceParallelModelPlugin(training_args.cp_mode)(model_engine.model, training_args.cp_size)

        input_ids = torch.arange(1, batch_size * 5 + 1, dtype=torch.long).view(batch_size, 5)
        model_inputs = {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(1, 6, dtype=torch.long).repeat(batch_size, 1),
            "loss_weights": torch.ones(batch_size, 5),
        }

        loss = sequence_parallel_loss(model_engine.model, model_inputs)
        assert loss is not None


@pytest.mark.runs_on(["cuda", "npu"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize(("cp_size", "dp_size", "batch_size"), [(2, 1, 1), (2, 1, 2)])
def test_sequence_parallel_loss(cp_size, dp_size, batch_size):
    master_port = find_available_port()
    world_size = cp_size * dp_size
    mp.spawn(
        _test_sequence_parallel_loss, args=(world_size, master_port, cp_size, dp_size, batch_size), nprocs=world_size
    )
