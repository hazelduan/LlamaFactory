# Copyright 2026 the LlamaFactory team.
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

"""Model-specific V1 adaptations; training and generic rendering stay in core."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..core.rendering import Renderer
    from ..utils.types import HFConfig, Processor


def create_renderer(processor: "Processor", config: "HFConfig", *, use_model_template: bool = True) -> "Renderer":
    """Select a renderer by model type, preserving explicit user template overrides."""
    from ..core.rendering import Renderer

    if use_model_template and getattr(config, "model_type", None) == "deepseek_v4":
        from .deepseek_v4 import DeepseekV4Renderer

        return DeepseekV4Renderer(processor)
    return Renderer(processor)
