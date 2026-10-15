import copy
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACompressor, DeepseekV4HCACompressor

from llamafactory.v1.plugins.model_plugins.parallelization.deepseek_v4 import _prepare_deepseek_v4_compressors


def worker(rank, path):
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
    for cls in (DeepseekV4CSACompressor, DeepseekV4HCACompressor):
        torch.manual_seed(42)
        config = DeepseekV4Config(
            hidden_size=32,
            head_dim=16,
            q_lora_rank=16,
            num_attention_heads=4,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=2,
            partial_rotary_factor=0.5,
            num_hidden_layers=2,
            compress_ratios=[4, 128],
        )
        ref = cls(config)
        for p in ref.parameters():
            torch.nn.init.normal_(p, mean=0.1, std=0.02)
        candidate = copy.deepcopy(ref)
        h = torch.randn(1, 256, 32, requires_grad=True)
        q = torch.randn(1, 256, 16, requires_grad=True)
        pos = torch.arange(256).unsqueeze(0)
        expected = ref(h, q, pos, None, 0)
        expected.square().sum().backward()
        localh = h.detach().chunk(2, dim=1)[rank].contiguous().requires_grad_()
        localq = q.detach().chunk(2, dim=1)[rank].contiguous().requires_grad_()
        _prepare_deepseek_v4_compressors(candidate, dist.group.WORLD)
        actual = candidate(localh, localq, pos.chunk(2, dim=1)[rank], None, 0)
        torch.testing.assert_close(actual, expected.detach().chunk(2, dim=2)[rank], atol=1e-6, rtol=1e-5)
        actual.square().sum().backward()
        torch.testing.assert_close(localh.grad, h.grad.chunk(2, dim=1)[rank], atol=1e-6, rtol=1e-5)
        for (name, p), (_, r) in zip(candidate.named_parameters(), ref.named_parameters()):
            if r.grad is None:
                assert p.grad is None, name
            else:
                dist.all_reduce(p.grad)
                torch.testing.assert_close(p.grad, r.grad, atol=1e-5, rtol=1e-4)
        if rank == 0:
            print("PASS native compressor output/input/parameter gradients", cls.__name__, flush=True)
    dist.destroy_process_group()


def test_native_compressors_preserve_full_sequence_forward_and_backward():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(worker, args=(d + "/pg",), nprocs=2, join=True)
