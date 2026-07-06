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

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from .....utils.plugin import BasePlugin


class EPDispatcherPlugin(BasePlugin):
    """Plugin registry for expert-parallel dispatcher forward builders."""

    def __call__(self, ep_group, **kwargs):
        return super().__call__(ep_group, **kwargs)


def get_dispatcher_factory(dispatcher: str | Callable):
    if isinstance(dispatcher, Callable):
        return lambda ep_group, fixed_router=False: partial(dispatcher, ep_group)

    def factory(ep_group, fixed_router=False):
        return EPDispatcherPlugin(dispatcher)(ep_group, fixed_router=fixed_router)

    return factory


def ensure_fsdp_turbo_importable() -> None:
    try:
        import fsdp_turbo  # noqa: F401

        return
    except ImportError:
        workspace_root = next(
            (p for p in Path(__file__).resolve().parents if (p / "FSDPTurbo" / "fsdp_turbo").exists()),
            None,
        )
        if workspace_root is None:
            raise ImportError(
                "FSDPTurbo is required for `dist_config.name: fsdp_turbo`. "
                "Please install fsdp-turbo or add the FSDPTurbo repository to PYTHONPATH."
            )

        fsdp_turbo_root = workspace_root / "FSDPTurbo"
        if str(fsdp_turbo_root) not in sys.path:
            sys.path.insert(0, str(fsdp_turbo_root))


@EPDispatcherPlugin("eager").register()
def build_eager_dispatcher(ep_group, fixed_router=False):
    ensure_fsdp_turbo_importable()

    from fsdp_turbo.distributed.expert_parallel.dispatcher import get_experts_forward_fn

    return get_experts_forward_fn(ep_group, fused=False, fixed_router=fixed_router)


@EPDispatcherPlugin("fused").register()
def build_fused_dispatcher(ep_group, fixed_router=False):
    ensure_fsdp_turbo_importable()

    from fsdp_turbo.distributed.expert_parallel.dispatcher import get_experts_forward_fn

    return get_experts_forward_fn(ep_group, fused=True, fixed_router=fixed_router)


@EPDispatcherPlugin("mc2").register()
def build_mc2_dispatcher(ep_group, fixed_router=False):
    ensure_fsdp_turbo_importable()

    from fsdp_turbo.distributed.expert_parallel.dispatcher_mc2 import get_experts_forward_mc2_fn

    return get_experts_forward_mc2_fn(ep_group, fixed_router=fixed_router)


@EPDispatcherPlugin("domino").register()
def build_domino_dispatcher(ep_group, fixed_router=False):
    ensure_fsdp_turbo_importable()

    from fsdp_turbo.distributed.expert_parallel.domino_dispatcher import get_domino_experts_forward_fn

    return get_domino_experts_forward_fn(ep_group, fixed_router=fixed_router)


def apply_hccl_premul_sum_patch() -> None:
    ensure_fsdp_turbo_importable()

    from fsdp_turbo.utils.torch_patch import apply_hccl_premul_sum_patch

    apply_hccl_premul_sum_patch()
