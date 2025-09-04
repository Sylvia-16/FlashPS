# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import (
    FluxTransformer2DLoadersMixin,
    FromOriginalModelMixin,
    PeftAdapterMixin,
)
from ...models.attention import FeedForward
from ...models.attention_processor import (
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)

# ### Import from attention_processor_short
from ...models.attention_processor_short import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
)
from ...models.modeling_utils import ModelMixin
from ...models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.import_utils import is_torch_npu_available
from ...utils.torch_utils import maybe_allow_in_graph
from ..cache_utils import CacheMixin
from ..embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    FluxPosEmbed,
)
from ..modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import time


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        block_index: int = -1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.block_name = f"flux_single_{block_index}"

        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if is_torch_npu_available():
            deprecation_message = (
                "Defaulting to FluxAttnProcessor2_0_NPU for NPU devices will be removed. Attention processors "
                "should be set explicitly using the `set_attn_processor` method."
            )
            deprecate("npu_processor", "0.34.0", deprecation_message)
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        edit_config: Optional[Dict[str, Any]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        norm_hidden_states, gate = self.norm(
            hidden_states,
            emb=temb,
            indices=edit_config.indices if edit_config is not None else None ,
        )
        assert gate.ndim == norm_hidden_states.ndim
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        if edit_config is not None:
            edit_config.block_name = self.block_name
            block_id = int(edit_config.block_name.split("_")[-1]) + 19
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            edit_config=edit_config,
            **joint_attention_kwargs,
        )
        if attn_output.ndim == 2:
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        else:
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)

        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        block_index: int = -1,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.block_name = f"flux_{block_index}"

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        edit_config: Optional[Dict[str, Any]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states,
            emb=temb,
            indices=edit_config.latents_indices if edit_config is not None else None,
        )

        assert (
            gate_msa.ndim == norm_hidden_states.ndim
        ), f"{gate_msa.shape} {norm_hidden_states.shape}"
        assert (
            gate_mlp.ndim == norm_hidden_states.ndim
        ), f"{gate_mlp.shape} {norm_hidden_states.shape}"
        assert (
            shift_mlp.ndim == norm_hidden_states.ndim
        ), f"{shift_mlp.shape} {norm_hidden_states.shape}"
        assert (
            scale_mlp.ndim == norm_hidden_states.ndim
        ), f"{scale_mlp.shape} {norm_hidden_states.shape}"

        (
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1_context(
            encoder_hidden_states,
            emb=temb,
            indices=edit_config.text_indices if edit_config is not None else None,
        )

        assert (
            c_gate_msa.ndim == norm_encoder_hidden_states.ndim
        ), f"{c_gate_msa.shape} {norm_encoder_hidden_states.shape}"
        assert (
            c_gate_mlp.ndim == norm_encoder_hidden_states.ndim
        ), f"{c_gate_mlp.shape} {norm_encoder_hidden_states.shape}"
        assert (
            c_shift_mlp.ndim == norm_encoder_hidden_states.ndim
        ), f"{c_shift_mlp.shape} {norm_encoder_hidden_states.shape}"
        assert (
            c_scale_mlp.ndim == norm_encoder_hidden_states.ndim
        ), f"{c_scale_mlp.shape} {norm_encoder_hidden_states.shape}"
        joint_attention_kwargs = joint_attention_kwargs or {}
        if edit_config is not None:
            edit_config.block_name = self.block_name
            block_id = int(edit_config.block_name.split("_")[-1])
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            edit_config=edit_config,
            **joint_attention_kwargs,
        )
        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output
        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        context_ff_output = c_gate_mlp * context_ff_output
        encoder_hidden_states = encoder_hidden_states + context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    block_index=i,
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    block_index=i,
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _copy_cache_buffers(
        self,
        cache_value_buffers,
        cache_key_buffers,
        cached_kv,
        batch_cache_map,
        layer_idx,
        block_name,
        denoising_step,
        batch_idx,
        non_blocking=True,
    ):
        # Copy value buffer for the current batch
        cache_value_buffers[layer_idx % 2][batch_idx : batch_idx + 1].copy_(
            cached_kv[f"v_{block_name}_{denoising_step}"][batch_cache_map[batch_idx]],
            non_blocking=non_blocking,
        )

        # Copy key buffer for the current batch
        cache_key_buffers[layer_idx % 2][batch_idx : batch_idx + 1].copy_(
            cached_kv[f"k_{block_name}_{denoising_step}"][batch_cache_map[batch_idx]],
            non_blocking=non_blocking,
        )

    def load_cache_per_layer(self, edit_config, layer_idx, is_single_layer):
        if edit_config is not None:
            denoising_step = edit_config.denoising_step
            batch_cache_map = edit_config.batch_cache_map
        if not is_single_layer:
            block_name = f"flux_{layer_idx+1}"
            # if is last layer, copy from single layer buffer
            if not (layer_idx == 0):
                self.compute_events[layer_idx - 1].wait()

            if layer_idx == 18:
                for batch_idx in range(edit_config.batch_size):
                    self._copy_cache_buffers(
                        self.cache_value_single_buffers,
                        self.cache_key_single_buffers,
                        edit_config.cached_kv,
                        batch_cache_map,
                        layer_idx + 1,
                        "flux_single_0",
                        denoising_step,
                        batch_idx,
                    )
            else:
                # copy from mmdit layer buffer
                for batch_idx in range(edit_config.batch_size):
                    self._copy_cache_buffers(
                        self.cache_value_buffers,
                        self.cache_key_buffers,
                        edit_config.cached_kv,
                        batch_cache_map,
                        layer_idx + 1,
                        block_name,
                        denoising_step,
                        batch_idx,
                    )
            self.loading_events[layer_idx + 1].record(
                stream=edit_config.load_stream
            )
        else:
            block_name = f"flux_single_{layer_idx+1}"
            event_idx = layer_idx + 19
            self.compute_events[event_idx - 1].wait()

            # if is last layer, copy from mmdit layer buffer for the next step
            if (
                layer_idx == 37
                and denoising_step != edit_config.num_inference_steps - 1
            ):
                for batch_idx in range(edit_config.batch_size):
                    self._copy_cache_buffers(
                        self.cache_value_buffers,
                        self.cache_key_buffers,
                        edit_config.cached_kv,
                        batch_cache_map,
                        layer_idx + 1,
                        "flux_0",
                        denoising_step + 1,
                        batch_idx,
                    )
                self.loading_events[0].record(stream=edit_config.load_stream)
                return
            elif (
                layer_idx == 37
                and denoising_step == edit_config.num_inference_steps - 1
            ):
                return
            # copy from single layer buffer for next layer in current step
            for batch_idx in range(edit_config.batch_size):
                self._copy_cache_buffers(
                    self.cache_value_single_buffers,
                    self.cache_key_single_buffers,
                    edit_config.cached_kv,
                    batch_cache_map,
                    layer_idx + 1,
                    block_name,
                    denoising_step,
                    batch_idx,
                )
                self.loading_events[event_idx + 1].record(
                    stream=edit_config.load_stream
                )

    def load_cache(self, edit_config):
        assert edit_config.async_copy, "async_copy should be enabled"
        denoising_step = edit_config.denoising_step
        # For first two steps, directly copy to GPU
        batch_cache_map = edit_config.batch_cache_map
        step_idx = (denoising_step) % 2
        # if denoising_step == 0 and layer_idx == 0, init

        for layer_idx in range(19):  # mmdit layer
            # next layer's block name
            block_name = f"flux_{layer_idx+1}"
            # if is last layer, copy from single layer buffer
            if not (layer_idx == 0 and denoising_step == 0):
                self.compute_events[layer_idx].wait()

            if layer_idx == 18:
                for batch_idx in range(edit_config.batch_size):
                    self._copy_cache_buffers(
                        self.cache_value_single_buffers,
                        self.cache_key_single_buffers,
                        edit_config.cached_kv,
                        batch_cache_map,
                        layer_idx + 1,
                        "flux_single_0",
                        denoising_step,
                        batch_idx,
                    )
            else:
                # copy from mmdit layer buffer
                for batch_idx in range(edit_config.batch_size):
                    self._copy_cache_buffers(
                        self.cache_value_buffers,
                        self.cache_key_buffers,
                        edit_config.cached_kv,
                        batch_cache_map,
                        layer_idx + 1,
                        block_name,
                        denoising_step,
                        batch_idx,
                    )
            self.loading_events[layer_idx + 1].record(
                stream=edit_config.load_stream
            )

        for layer_idx in range(38):  # layer
            block_name = f"flux_single_{layer_idx+1}"
            event_idx = layer_idx + 19
            self.compute_events[event_idx].wait()

            # if is last layer, copy from mmdit layer buffer for the next step
            if (
                layer_idx == 37
                and denoising_step != edit_config.num_inference_steps - 1
            ):
                for batch_idx in range(edit_config.batch_size):
                    self._copy_cache_buffers(
                        self.cache_value_buffers,
                        self.cache_key_buffers,
                        edit_config.cached_kv,
                        batch_cache_map,
                        layer_idx + 1,
                        "flux_0",
                        denoising_step + 1,
                        batch_idx,
                    )
                self.loading_events[0].record(stream=edit_config.load_stream)
                break
            elif (
                layer_idx == 37
                and denoising_step == edit_config.num_inference_steps - 1
            ):
                break
            # copy from single layer buffer for next layer in current step
            for batch_idx in range(edit_config.batch_size):
                self._copy_cache_buffers(
                    self.cache_value_single_buffers,
                    self.cache_key_single_buffers,
                    edit_config.cached_kv,
                    batch_cache_map,
                    layer_idx + 1,
                    block_name,
                    denoising_step,
                    batch_idx,
                )
                self.loading_events[event_idx + 1].record(
                    stream=edit_config.load_stream
                )

    def init_cache(self, edit_config):
        denoising_step = edit_config.denoising_step
        if denoising_step == 0 and not hasattr(self, "loading_events"):
            print("init cache")
            self.cat_done_event = torch.cuda.Event(enable_timing=False)
            # Initialize buffers for ping-pong caching
            self.loading_events = [
                torch.cuda.Event(edit_config.device_num) for _ in range(57)
            ]
            self.compute_events = [
                torch.cuda.Event(edit_config.device_num) for _ in range(57)
            ]

            def _get_single_buffer(
                edit_config, block_name, denoising_step, is_value=True
            ):
                cached_kv_key_name = (
                    f"v_{block_name}_{denoising_step}"
                    if is_value
                    else f"k_{block_name}_{denoising_step}"
                )
                single_buffer = torch.zeros_like(
                    edit_config.cached_kv[cached_kv_key_name][0],
                    dtype=edit_config.cached_kv[cached_kv_key_name][0].dtype,
                ).cuda(edit_config.device_num)
                # if edit_config.batch_size > 1:
                # max batch size
                if hasattr(edit_config, "max_batch_size"):
                    max_batch_size = edit_config.max_batch_size
                else:
                    max_batch_size = edit_config.batch_size
                single_buffer = single_buffer.repeat(
                    max_batch_size, 1, 1
                ).contiguous()
                return single_buffer

            # mmdit layer buffer
            value_buffer = [
                _get_single_buffer(edit_config, "flux_0", denoising_step, is_value=True)
                for _ in range(2)
            ]
            key_buffer = [
                _get_single_buffer(
                    edit_config, "flux_0", denoising_step, is_value=False
                )
                for _ in range(2)
            ]

            self.cache_value_buffers = value_buffer
            self.cache_key_buffers = key_buffer
            # single layer buffer
            value_single_buffer = [
                _get_single_buffer(
                    edit_config, "flux_single_0", denoising_step, is_value=True
                )
                for _ in range(2)
            ]
            key_single_buffer = [
                _get_single_buffer(
                    edit_config, "flux_single_0", denoising_step, is_value=False
                )
                for _ in range(2)
            ]
            self.cache_value_single_buffers = value_single_buffer
            self.cache_key_single_buffers = key_single_buffer

        layer_idx = 0
        # load layer 0 and 1
        for batch_idx in range(edit_config.batch_size):
            self._copy_cache_buffers(
                self.cache_value_buffers,
                self.cache_key_buffers,
                edit_config.cached_kv,
                edit_config.batch_cache_map,
                layer_idx,
                "flux_0",
                denoising_step,
                batch_idx,
                non_blocking=False,
            )

    def compute_per_layer(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_blocks_repeat,
        controlnet_single_block_samples,
        edit_config,
    ):
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                    )
                )
            else:
                with torch.cuda.stream(edit_config.load_stream):
                    self.load_cache_per_layer(edit_config, index_block, False)
                with torch.cuda.stream(edit_config.compute_stream):
                    if edit_config.async_copy:
                        layer_idx = index_block % 2
                    if not (edit_config.denoising_step == 0 and index_block == 0):
                        self.loading_events[index_block].wait()
                    edit_config.current_key_cache = self.cache_key_buffers[layer_idx][:edit_config.batch_size]
                    edit_config.current_value_cache = self.cache_value_buffers[
                        layer_idx
                    ][:edit_config.batch_size]

                    if edit_config.test_varlen:
                        edit_config.current_key_cache = (
                            edit_config.current_key_cache.flatten(0, 1).contiguous()
                        )
                        edit_config.current_value_cache = (
                            edit_config.current_value_cache.flatten(0, 1).contiguous()
                        )
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        edit_config=edit_config,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                    if edit_config.async_copy:
                        self.compute_events[index_block].record(
                            stream=edit_config.compute_stream
                        )

                if edit_config.async_copy:
                    self.compute_events[index_block].wait()
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(
                    controlnet_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[
                            index_block % len(controlnet_block_samples)
                        ]
                    )
                else:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block // interval_control]
                    )

        if hasattr(edit_config, "origin_hidden_states"):
            # hidden states shape: [batch_size*seqlen, hidden_dim]
            if hidden_states.ndim == 3:
                assert not hasattr(edit_config, "origin_hidden_states_shape")
            edit_config.origin_hidden_states[edit_config.test_target_indice] = (
                hidden_states
            )
            edit_config.origin_hidden_states[edit_config.test_target_text_indice] = (
                encoder_hidden_states
            )
            hidden_states = edit_config.origin_hidden_states
        else:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 创建一个事件来标记cat操作完成

        # 在当前默认流上记录事件
        self.cat_done_event.record()

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                with torch.cuda.stream(edit_config.load_stream):
                    self.cat_done_event.wait()
                    self.load_cache_per_layer(edit_config, index_block, True)
                with torch.cuda.stream(edit_config.compute_stream):
                    if edit_config.async_copy:
                        layer_idx = index_block % 2
                    event_idx = index_block + 19
                    self.cat_done_event.wait()
                    self.loading_events[event_idx].wait()
                    edit_config.current_key_cache = self.cache_key_single_buffers[
                        layer_idx
                    ][:edit_config.batch_size]
                    edit_config.current_value_cache = self.cache_value_single_buffers[
                        layer_idx
                    ][:edit_config.batch_size]
                    if edit_config.test_varlen:
                        edit_config.current_key_cache = (
                            edit_config.current_key_cache.flatten(0, 1).contiguous()
                        )
                        edit_config.current_value_cache = (
                            edit_config.current_value_cache.flatten(0, 1).contiguous()
                        )
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        edit_config=edit_config,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                    if edit_config.async_copy:
                        self.compute_events[event_idx].record(
                            stream=edit_config.compute_stream
                        )

                if edit_config.async_copy:
                    self.compute_events[index_block + 19].wait()

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        return hidden_states

    def compute(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_blocks_repeat,
        controlnet_single_block_samples,
        edit_config,
    ):
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                    )
                )
            else:
                if edit_config is not None and edit_config.async_copy:
                    layer_idx = index_block % 2
                    if not (edit_config.denoising_step == 0 and index_block == 0):
                        self.loading_events[index_block].wait()
                    edit_config.current_key_cache = self.cache_key_buffers[layer_idx]
                    edit_config.current_value_cache = self.cache_value_buffers[
                        layer_idx
                    ]

                    if edit_config.test_varlen:
                        edit_config.current_key_cache = (
                            edit_config.current_key_cache.flatten(0, 1).contiguous()
                        )
                        edit_config.current_value_cache = (
                            edit_config.current_value_cache.flatten(0, 1).contiguous()
                        )
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    edit_config=edit_config,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                if edit_config.async_copy:
                    self.compute_events[index_block].record(
                        stream=edit_config.compute_stream
                    )
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(
                    controlnet_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[
                            index_block % len(controlnet_block_samples)
                        ]
                    )
                else:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block // interval_control]
                    )
        if hasattr(edit_config, "origin_hidden_states"):
            # hidden states shape: [batch_size*seqlen, hidden_dim]
            if hidden_states.ndim == 3:
                assert not hasattr(edit_config, "origin_hidden_states_shape")
            edit_config.origin_hidden_states[edit_config.test_target_indice] = (
                hidden_states
            )
            edit_config.origin_hidden_states[edit_config.test_target_text_indice] = (
                encoder_hidden_states
            )
            hidden_states = edit_config.origin_hidden_states
        else:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                if edit_config is not None and edit_config.async_copy:
                    layer_idx = index_block % 2
                    event_idx = layer_idx + 19
                    self.loading_events[event_idx].wait()
                    edit_config.current_key_cache = self.cache_key_single_buffers[
                        layer_idx
                    ]
                    edit_config.current_value_cache = self.cache_value_single_buffers[
                        layer_idx
                    ]
                    if edit_config.test_varlen:
                        edit_config.current_key_cache = (
                            edit_config.current_key_cache.flatten(0, 1).contiguous()
                        )
                        edit_config.current_value_cache = (
                            edit_config.current_value_cache.flatten(0, 1).contiguous()
                        )
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    edit_config=edit_config,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                if edit_config.async_copy:
                    self.compute_events[event_idx].record(
                        stream=edit_config.compute_stream
                    )
            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
        return hidden_states

    def cal_with_async_copy(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_blocks_repeat,
        controlnet_single_block_samples,
        edit_config,
    ):
        # assert edit_config.async_copy, "async_copy should be enabled"
        # Load cache in load stream
        denoising_step = edit_config.denoising_step
        with torch.cuda.stream(edit_config.load_stream):
            self.load_cache(edit_config)
        with torch.cuda.stream(edit_config.compute_stream):
            # Wait for load to complete if past first step
            # 在继续处理之前，确保复制完成
            # Compute attention
            hidden_states = self.compute(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                controlnet_single_block_samples=controlnet_single_block_samples,
                edit_config=edit_config,
            )
        self.compute_events[56].wait()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        edit_config: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        # print("timestep",timestep)
        # print("guidance",guidance)
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        batch_size = hidden_states.shape[0]
        if edit_config is not None and edit_config.test_varlen:
            cos = image_rotary_emb[0].repeat(batch_size, 1, 1).flatten(0, 1)
            sin = image_rotary_emb[1].repeat(batch_size, 1, 1).flatten(0, 1)
        else:
            cos = image_rotary_emb[0]
            sin = image_rotary_emb[1]

        cos = cos[:, ::2].contiguous()
        sin = sin[:, ::2].contiguous()

        if edit_config is not None and edit_config.use_cached_o or edit_config.use_cached_kv:
            edit_config.cos_q = cos[edit_config.mask_indice].contiguous()
            edit_config.sin_q = sin[edit_config.mask_indice].contiguous()
        if edit_config is not None:
            edit_config.cos = cos
            edit_config.sin = sin

        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop(
                "ip_adapter_image_embeds"
            )
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
      
        if edit_config is not None and edit_config.use_cached_kv:
            if edit_config.test_varlen:
                # 把shape0和shape1合并为一个shape
                
                hidden_states = hidden_states.view(
                    hidden_states.shape[0] * hidden_states.shape[1], -1
                )
                hidden_states = hidden_states[
                    edit_config.latents_mask_indice, :
                ].contiguous()  # only two dimension, [seqlen*batch_size, hidden_dim]
                encoder_hidden_states = encoder_hidden_states.view(
                    encoder_hidden_states.shape[0] * encoder_hidden_states.shape[1], -1
                )
            else:
                latents_mask_indice = edit_config.latents_mask_indice
                hidden_states = hidden_states[
                    :, latents_mask_indice, :
                ].contiguous()
        if edit_config is not None and edit_config.async_copy:
            if edit_config.denoising_step == 0:
                self.init_cache(edit_config)
            # hidden_states = self.cal_with_async_copy(
            #     hidden_states=hidden_states,
            #     encoder_hidden_states=encoder_hidden_states,
            #     temb=temb,
            #     image_rotary_emb=image_rotary_emb,
            #     joint_attention_kwargs=joint_attention_kwargs,
            #     controlnet_block_samples=controlnet_block_samples,
            #     controlnet_blocks_repeat=controlnet_blocks_repeat,
            #     controlnet_single_block_samples=controlnet_single_block_samples,
            #     edit_config=edit_config,
            # )
            hidden_states = self.compute_per_layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                controlnet_single_block_samples=controlnet_single_block_samples,
                edit_config=edit_config,
            )

        else:
            hidden_states = self.compute(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                controlnet_single_block_samples=controlnet_single_block_samples,
                edit_config=edit_config,
            )
        # 从hidden_states中提取latent部分
        if hidden_states.ndim == 2:
            if edit_config.denoising_step == 0:
                self.latents_hidden_states = torch.zeros(
                    (edit_config.latents_mask_indice.shape[0], hidden_states.shape[1]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            hidden_states = hidden_states[edit_config.test_target_indice]
        else:
            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(
            hidden_states, temb, indices=edit_config.latents_indices
        )
        output = self.proj_out(hidden_states)
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
