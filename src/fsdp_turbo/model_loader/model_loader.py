# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import os
import glob
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, Set, Callable

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights

from fsdp_turbo.model_loader.device_manager import get_device_type
import logging

logger = logging.getLogger(__name__)


@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device.
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = (
                param
                if param.device == torch.device("meta")
                else param_cls(module._parameters[name].to("meta"), **kwargs)
            )
    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


def reset_hf_initialized_flag(module: nn.Module) -> None:
    """Reset HuggingFace's _is_hf_initialized flag."""
    if hasattr(module, "_is_hf_initialized"):
        setattr(module, "_is_hf_initialized", False)
    for child in module.children():
        reset_hf_initialized_flag(child)


def _find_submodule(module: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Find the leaf module according to the name."""
    pieces = name.split(".")
    for piece in pieces[:-1]:
        module = getattr(module, piece)
    return module, pieces[-1]


class ModelLoader:
    """Load model on CPU or meta device."""

    def __init__(self, model_path: str, trust_remote_code: bool = False,
                 train_from_scratch: bool = False, init_device: str = "cpu",
                 custom_config: Optional[Dict] = None):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.train_from_scratch = train_from_scratch
        self.init_device = init_device
        self.custom_config = custom_config or {}
        self.hf_config = None

    def load_config(self) -> AutoConfig:
        """Load HuggingFace model config."""
        logger.info(f"> Loading config from {self.model_path}...")
        self.hf_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        for key, value in self.custom_config.items():
            if hasattr(self.hf_config, key):
                setattr(self.hf_config, key, value)
                logger.info(f"> Overrode config: {key}={value}")
        return self.hf_config

    def create_model(self, model_cls=None) -> Tuple[nn.Module, Optional[str]]:
        """Create model based on init_device."""
        if self.hf_config is None:
            self.load_config()

        if self.init_device == "meta":
            return self._create_on_meta(model_cls)
        else:
            return self._create_on_cpu(model_cls)

    def _create_on_cpu(self, model_cls=None) -> Tuple[nn.Module, None]:
        """Create and load model on CPU."""
        if model_cls is not None:
            logger.info(f"> Loading {model_cls.__name__} on CPU...")
            model = model_cls.from_pretrained(
                self.model_path,
                config=self.hf_config,
                low_cpu_mem_usage=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
        elif self.train_from_scratch:
            logger.info("> Creating model with random weights on CPU...")
            model = AutoModelForCausalLM.from_config(
                self.hf_config,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float32
            )
        else:
            logger.info(f"> Loading pretrained model on CPU from {self.model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.hf_config,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )

        return model, None

    def _create_on_meta(self, model_cls=None) -> Tuple[nn.Module, Optional[str]]:
        """Create empty model on meta device."""
        weights_path = None if self.train_from_scratch else self.model_path

        if model_cls is not None:
            logger.info(f"> Creating empty {model_cls.__name__} on meta device...")
            with init_empty_weights(), no_init_weights():
                if hasattr(model_cls, '_from_config'):
                    model = model_cls._from_config(self.hf_config)
                else:
                    model = model_cls.from_config(self.hf_config)
        elif self.train_from_scratch:
            logger.info("> Creating empty model on meta device for random init...")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(
                    self.hf_config,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float32
                )
        else:
            logger.info(f"> Creating empty model on meta device (weights: {self.model_path})...")
            with init_empty_weights(), no_init_weights():
                model = AutoModelForCausalLM.from_config(
                    self.hf_config,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float32
                )

        logger.info(f"> Model created on meta device. Weights path: {weights_path}")
        return model, weights_path


class WeightLoader:
    """
    Load weights into FSDP-wrapped model.
    """

    @staticmethod
    def load(
        model: nn.Module,
        weights_path: Optional[str],
        device: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:
        """Load or initialize weights after FSDP wrapping."""
        if device is None:
            device = get_device_type()

        if weights_path is None:
            WeightLoader._init_random(model, device, seed)
        else:
            WeightLoader._load_pretrained(model, weights_path, device, seed)

    @staticmethod
    def preload_metadata(weights_path: str) -> Dict[str, Dict]:
        """
        Pre-scan checkpoint files and return metadata without loading full tensors into memory.

        Returns:
            Dict mapping parameter name -> {"file": str, "key": str, "dtype": torch.dtype, "shape": tuple}
        """
        logger.info(f"> Pre-scanning checkpoint metadata from {weights_path}...")
        metadata = {}
        state_dict_files = WeightLoader._get_state_dict_files(weights_path)

        for filepath in state_dict_files:
            if filepath.endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        metadata[key] = {
                            "file": filepath,
                            "key": key,
                            "dtype": tensor.dtype,
                            "shape": tuple(tensor.shape),
                        }
            else:
                state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
                for key, tensor in state_dict.items():
                    metadata[key] = {
                        "file": filepath,
                        "key": key,
                        "dtype": tensor.dtype,
                        "shape": tuple(tensor.shape),
                    }

        logger.info(f"> Pre-scanned {len(metadata)} parameter keys from checkpoint")
        return metadata

    @staticmethod
    def create_param_init_fn(
        model: nn.Module,
        metadata: Dict[str, Dict],
        device: str,
        seed: Optional[int] = None,
    ) -> Callable[[nn.Module], None]:
        """
        Create a param_init_fn for FSDP's lazy initialization.

        This function is called by FSDP for each FSDP unit during fully_shard().
        It materializes the module's parameters directly to the target device and
        loads the corresponding weights from the checkpoint.

        This is the industry-standard approach: parameters are never materialized
        on CPU; they go directly from meta device to the target accelerator device
        one FSDP unit at a time, minimizing peak memory usage.
        """
        module_prefix = {}
        for full_name, mod in model.named_modules():
            module_prefix[mod] = full_name

        metadata_keys = set(metadata.keys())

        def param_init_fn(module: nn.Module) -> None:
            any_meta = any(
                p.device.type == 'meta' for p in module.parameters()
            )
            if not any_meta:
                return

            prefix = module_prefix.get(module, "")
            logger.debug(f"> [param_init_fn] Materializing module '{prefix}' to {device}")

            module.to_empty(device=device)

            loaded_count = 0
            for local_name, param in module.named_parameters():
                full_name = f"{prefix}.{local_name}" if prefix else local_name
                if full_name in metadata_keys:
                    info = metadata[full_name]
                    weight = WeightLoader._load_single_tensor(info)
                    param.data.copy_(weight.to(device=param.device, dtype=param.dtype))
                    loaded_count += 1

            if loaded_count > 0:
                logger.debug(
                    f"> [param_init_fn] Loaded {loaded_count} weights for '{prefix}'"
                )

        return param_init_fn

    @staticmethod
    def create_random_init_fn(
        model: nn.Module,
        device: str,
        seed: Optional[int] = None,
    ) -> Callable[[nn.Module], None]:
        """
        Create a param_init_fn for random initialization during FSDP wrapping.

        This is the industry-standard approach for training from scratch with
        meta device: each FSDP unit is materialized and randomly initialized
        one at a time via param_init_fn, never materializing the full model on CPU.
        """
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.accelerator.is_available():
                torch.accelerator.manual_seed(seed)
                torch.accelerator.manual_seed_all(seed)
            logger.info(f"> Set seed to {seed} for random initialization")

        def param_init_fn(module: nn.Module) -> None:
            any_meta = any(
                p.device.type == 'meta' for p in module.parameters()
            )
            if not any_meta:
                return

            logger.debug(f"> [random_init_fn] Materializing module to {device}")
            module.to_empty(device=device)
            reset_hf_initialized_flag(module)
            if hasattr(module, 'init_weights'):
                module.init_weights()

        return param_init_fn

    @staticmethod
    def _load_single_tensor(info: Dict) -> torch.Tensor:
        """Load a single tensor from a checkpoint file."""
        filepath = info["file"]
        key = info["key"]
        if filepath.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(filepath, framework="pt", device="cpu") as f:
                return f.get_tensor(key)
        else:
            state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
            return state_dict[key]

    @staticmethod
    def _init_random(model: nn.Module, device: str, seed: Optional[int] = None) -> None:
        """Initialize model with random weights."""
        logger.info(f"> Initializing random weights on {device}...")

        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.accelerator.is_available():
                torch.accelerator.manual_seed(seed)
                torch.accelerator.manual_seed_all(seed)
            logger.info(f"> Set seed to {seed} for random initialization")

        model.to_empty(device=device)
        reset_hf_initialized_flag(model)

        if hasattr(model, 'init_weights'):
            model.init_weights()

        logger.info("> Random initialization done")

    @staticmethod
    @torch.no_grad()
    def _load_pretrained(model: nn.Module, weights_path: str, device: str, seed: Optional[int] = None) -> None:
        """
        Load pretrained weights BEFORE any parallel wrapping.

        Since no TP/EP/FSDP has been applied yet, all parameters are plain tensors.
        We materialize the model to CPU (so FSDP handles device transfer, same as
        the non-meta path) and copy checkpoint weights directly.

        CRITICAL: We save buffer values BEFORE to_empty() because to_empty()
        replaces ALL tensors (including buffers) with empty tensors, destroying
        buffer values that were correctly computed during __init__().
        """
        import random as python_random
        import numpy as np

        if seed is not None:
            python_random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.accelerator.is_available():
                torch.accelerator.manual_seed(seed)
                torch.accelerator.manual_seed_all(seed)
            logger.info(f"> Set seed to {seed} before loading weights")

        logger.info(f"> Loading pretrained weights from {weights_path}...")

        original_dtypes = {}
        for name, param in model.named_parameters():
            original_dtypes[name] = param.dtype

        parameter_names_to_load = {name for name, _ in model.named_parameters()}
        logger.info(f"> {len(parameter_names_to_load)} parameters to load")

        buffer_backup = {}
        for name, buffer in model.named_buffers():
            if buffer.device.type != "meta":
                buffer_backup[name] = buffer.clone()
        logger.info(f"> Saved {len(buffer_backup)} buffers before materialization")

        model.to_empty(device=device)
        logger.info(f"> Model materialized to {device}")

        for name, buffer in buffer_backup.items():
            try:
                module, local_name = _find_submodule(model, name)
                model_buffer = dict(module.named_buffers(recurse=False))[local_name]
                model_buffer.copy_(buffer.to(device=model_buffer.device, dtype=model_buffer.dtype))
            except Exception as e:
                logger.warning(f"> Failed to restore buffer {name}: {e}")
        logger.info(f"> Restored {len(buffer_backup)} buffers after materialization")

        buffer_dict = {}
        for name, buffer in model.named_buffers():
            buffer_dict[name] = buffer.clone()
        logger.info(f"> Snapshot {len(buffer_dict)} buffers for checkpoint overlay")

        state_dict_files = WeightLoader._get_state_dict_files(weights_path)

        loaded_count = 0
        for state_dict_file in state_dict_files:
            for name, tensor in WeightLoader._iterate_state_dict(state_dict_file):
                if name in buffer_dict:
                    buffer_dict[name] = tensor.clone()
                elif name in parameter_names_to_load:
                    parameter_names_to_load.remove(name)
                    target_dtype = original_dtypes.get(name, tensor.dtype)
                    WeightLoader._dispatch_parameter(model, name, tensor, target_dtype)
                    loaded_count += 1
                else:
                    logger.debug(f"> Unexpected key in state dict: {name}")

        logger.info(f"> Loaded {loaded_count} parameters from checkpoint")

        WeightLoader._post_process(model, buffer_dict, parameter_names_to_load, seed)

        logger.info("> Pretrained weights loaded successfully")

    @staticmethod
    def _get_state_dict_files(weights_path: str):
        """Get list of state dict files."""
        index_file = os.path.join(weights_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json
            with open(index_file, 'r') as f:
                index = json.load(f)
            files = set(index["weight_map"].values())
            return [os.path.join(weights_path, f) for f in sorted(files)]

        single_safetensor = os.path.join(weights_path, "model.safetensors")
        if os.path.exists(single_safetensor):
            return [single_safetensor]

        safetensor_files = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
        if safetensor_files:
            return safetensor_files

        pytorch_files = sorted(glob.glob(os.path.join(weights_path, "*.bin")))
        if pytorch_files:
            return pytorch_files

        pytorch_files = sorted(glob.glob(os.path.join(weights_path, "*.pt")))
        if pytorch_files:
            return pytorch_files

        raise FileNotFoundError(f"No weight files found in {weights_path}")

    @staticmethod
    def _iterate_state_dict(filepath: str):
        """Iterate over state dict file, yielding (key, tensor) pairs."""
        if filepath.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
        else:
            state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
            for key, tensor in state_dict.items():
                yield key, tensor

    @staticmethod
    def _dispatch_parameter(
        model: nn.Module,
        name: str,
        tensor: torch.Tensor,
        target_dtype: Optional[torch.dtype] = None
    ) -> None:
        """
        Assign parameter to model. No DTensor handling needed since
        weight loading happens before any parallel wrapping.
        """
        module, local_name = _find_submodule(model, name)
        param = dict(module.named_parameters(recurse=False))[local_name]
        dtype = target_dtype if target_dtype is not None else param.dtype
        param.data.copy_(tensor.to(device=param.device, dtype=dtype))

    @staticmethod
    def _dispatch_buffer(
        model: nn.Module,
        name: str,
        buffer: torch.Tensor,
    ) -> None:
        """
        Assign buffer to model. No DTensor handling needed since
        weight loading happens before any parallel wrapping.
        """
        module, local_name = _find_submodule(model, name)
        orig_buffer = dict(module.named_buffers(recurse=False))[local_name]
        orig_buffer.copy_(buffer.to(device=orig_buffer.device, dtype=orig_buffer.dtype))

    @staticmethod
    def _init_parameter(model: nn.Module, name: str, seed: Optional[int] = None, param_index: int = 0) -> None:
        """
        Initialize a single missing parameter deterministically.
        Only initializes the specific parameter, not the entire module.
        """
        import random as python_random
        import numpy as np

        pieces = name.split(".")
        module = model
        for piece in pieces[:-1]:
            module = getattr(module, piece)

        param_name = pieces[-1]
        param = getattr(module, param_name, None)

        if param is None or not isinstance(param, nn.Parameter):
            logger.warning(f"> Cannot find parameter {name}, skipping initialization")
            return

        if seed is not None:
            python_random.seed(seed + param_index)
            np.random.seed(seed + param_index)
            torch.manual_seed(seed + param_index)
            if torch.accelerator.is_available():
                torch.accelerator.manual_seed(seed + param_index)

        init_func = getattr(module, "_init_weights", None)
        if init_func is not None:
            init_func(module)
        else:
            nn.init.zeros_(param.data)
            logger.warning(
                f"> No _init_weights for {name}, initialized to zeros "
                f"to ensure deterministic behavior"
            )

    @staticmethod
    def _post_process(
        model: nn.Module,
        buffer_dict: Dict[str, torch.Tensor],
        parameter_names_left: Set[str],
        seed: Optional[int] = None
    ) -> None:
        """
        Post-process after weight loading: restore buffers, init missing params, tie embeddings.
        """
        buffer_failures = []
        for name, buffer in buffer_dict.items():
            try:
                WeightLoader._dispatch_buffer(model, name, buffer)
            except Exception as e:
                buffer_failures.append(name)
                logger.error(f"> Failed to restore buffer {name}: {e}")

        if buffer_failures:
            raise RuntimeError(
                f"Failed to restore {len(buffer_failures)} buffers: {buffer_failures}. "
                f"Buffer restoration is critical for model correctness."
            )
        logger.info(f"> Restored {len(buffer_dict)} buffers")

        if parameter_names_left:
            logger.warning(f"> Missing {len(parameter_names_left)} parameters, initializing them with seed {seed}...")
            for idx, name in enumerate(sorted(parameter_names_left)):
                try:
                    WeightLoader._init_parameter(model, name, seed, param_index=idx)
                    logger.warning(f"> Initialized missing parameter: {name}")
                except Exception as e:
                    logger.warning(f"> Failed to initialize {name}: {e}")

        if getattr(model.config, "tie_word_embeddings", True):
            try:
                input_embeddings = model.get_input_embeddings()
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is not None and input_embeddings is not None:
                    output_embeddings.weight = input_embeddings.weight
                    logger.info("> Tied input/output embeddings")
            except Exception as e:
                logger.warning(f"> Failed to tie embeddings: {e}")
