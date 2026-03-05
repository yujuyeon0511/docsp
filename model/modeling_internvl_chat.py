# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import torch.nn.functional as F
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .docsp_projector import DocSPProjector

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.spatial_feature_mode = getattr(config, "spatial_feature_mode", "none")
        self.spatial_pool_size = self._normalize_hw(
            getattr(config, "spatial_pool_size", None), "spatial_pool_size"
        )
        self.spatial_crop_size = self._normalize_hw(
            getattr(config, "spatial_crop_size", None), "spatial_crop_size"
        )
        self.spatial_crop_stride = self._normalize_hw(
            getattr(config, "spatial_crop_stride", None), "spatial_crop_stride"
        )
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        base_grid_size = int((image_size // patch_size) * self.downsample_ratio)
        base_num_image_token = base_grid_size * base_grid_size
        extra_tokens_per_image = self._infer_extra_tokens(base_grid_size)
        self.num_image_token = base_num_image_token + extra_tokens_per_image

        logger.info(f'num_image_token: {self.num_image_token}')
        if extra_tokens_per_image > 0:
            logger.info(
                f'spatial_feature_mode: {self.spatial_feature_mode}, '
                f'extra_tokens_per_image: {extra_tokens_per_image}'
            )
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if getattr(config, "llm_use_pretrained", False):
                if not config.llm_model_name_or_path:
                    raise ValueError("llm_use_pretrained=True requires llm_model_name_or_path")
                try:
                    from transformers import modeling_rope_utils
                    if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
                        modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = (
                            modeling_rope_utils._compute_default_rope_parameters
                        )
                except Exception:
                    # Best-effort fallback; if the mapping is missing in older versions,
                    # this prevents a KeyError for rope_type="default".
                    pass
                llm_config = AutoConfig.from_pretrained(
                    config.llm_model_name_or_path,
                    trust_remote_code=getattr(config, "llm_trust_remote_code", False),
                )
                # HyperCLOVA models may store rope_scaling/rope_type as None or "default".
                # Force linear scaling with factor=1.0 to avoid ROPE_INIT_FUNCTIONS["default"] issues.
                rope_scaling = getattr(llm_config, "rope_scaling", None)
                rope_scaling_type = None
                if isinstance(rope_scaling, dict):
                    rope_scaling_type = rope_scaling.get("rope_type")
                elif rope_scaling is not None:
                    rope_scaling_type = getattr(rope_scaling, "rope_type", None)
                if rope_scaling is None or rope_scaling_type == "default":
                    llm_config.rope_scaling = {"rope_type": "linear", "factor": 1.0}
                self.language_model = AutoModelForCausalLM.from_pretrained(
                    config.llm_model_name_or_path,
                    trust_remote_code=getattr(config, "llm_trust_remote_code", False),
                    config=llm_config,
                    torch_dtype=getattr(llm_config, "torch_dtype", None),
                    low_cpu_mem_usage=True,
                )
                config.llm_config = llm_config
            else:
                architecture: str = config.llm_config.architectures[0]
                if architecture == 'LlamaForCausalLM':
                    self.language_model = LlamaForCausalLM(config.llm_config)
                elif architecture == 'Qwen2ForCausalLM':
                    self.language_model = Qwen2ForCausalLM(config.llm_config)
                elif architecture == 'Qwen3MoeForCausalLM':
                    self.language_model = Qwen3MoeForCausalLM(config.llm_config)
                elif architecture == 'Qwen3ForCausalLM':
                    self.language_model = Qwen3ForCausalLM(config.llm_config)
                else:
                    raise NotImplementedError(f'{architecture} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # DocSP: Document Structure-aware Spatial Perception Projector
        self.use_docsp = getattr(config, 'use_docsp', False)
        if self.use_docsp:
            docsp_grid = image_size // patch_size  # pre-pixel-shuffle grid (e.g., 32)
            self.docsp = DocSPProjector(
                vision_dim=vit_hidden_size,
                llm_dim=llm_hidden_size,
                grid_size=docsp_grid,
                mid_dim=getattr(config, 'docsp_mid_dim', 256),
                num_struct_tokens=getattr(config, 'docsp_num_struct_tokens', 4),
                num_sem_tokens=getattr(config, 'docsp_num_sem_tokens', 4),
                num_dfi_heads=getattr(config, 'docsp_num_dfi_heads', 4),
            )
            self.num_image_token += self.docsp.num_spatial_tokens
            logger.info(f'DocSP enabled: {self.docsp.num_spatial_tokens} spatial tokens, '
                        f'total num_image_token: {self.num_image_token}')

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def _normalize_hw(self, value, name: str):
        if value is None:
            return None
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(f"{name} must be an int or a tuple/list of length 2.")

    def _infer_extra_tokens(self, grid_size: int) -> int:
        if self.spatial_feature_mode == "none":
            return 0
        if self.spatial_feature_mode == "pool":
            pool_size = self.spatial_pool_size or (2, 2)
            return int(pool_size[0] * pool_size[1])
        if self.spatial_feature_mode == "crop":
            crop_size = self.spatial_crop_size or (2, 2)
            crop_stride = self.spatial_crop_stride or crop_size
            crop_h = min(int(crop_size[0]), grid_size)
            crop_w = min(int(crop_size[1]), grid_size)
            stride_h = max(1, min(int(crop_stride[0]), crop_h))
            stride_w = max(1, min(int(crop_stride[1]), crop_w))
            num_h = 1 + (grid_size - crop_h) // stride_h
            num_w = 1 + (grid_size - crop_w) // stride_w
            return int(num_h * num_w)
        if self.spatial_feature_mode == "pool_crop":
            pool_size = self.spatial_pool_size or (2, 2)
            pool_tokens = int(pool_size[0] * pool_size[1])
            crop_size = self.spatial_crop_size or (2, 2)
            crop_stride = self.spatial_crop_stride or crop_size
            crop_h = min(int(crop_size[0]), grid_size)
            crop_w = min(int(crop_size[1]), grid_size)
            stride_h = max(1, min(int(crop_stride[0]), crop_h))
            stride_w = max(1, min(int(crop_stride[1]), crop_w))
            num_h = 1 + (grid_size - crop_h) // stride_h
            num_w = 1 + (grid_size - crop_w) // stride_w
            crop_tokens = int(num_h * num_w)
            return pool_tokens + crop_tokens
        raise ValueError(f"Unsupported spatial_feature_mode: {self.spatial_feature_mode}")

    def _extract_spatial_tokens(self, grid: torch.Tensor) -> Optional[torch.Tensor]:
        if self.spatial_feature_mode == "none":
            return None

        features = grid.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        if self.spatial_feature_mode == "pool":
            pool_size = self.spatial_pool_size or (2, 2)
            pooled = F.adaptive_avg_pool2d(features, pool_size)
            return pooled.flatten(2).transpose(1, 2)
        if self.spatial_feature_mode == "crop":
            crop_size = self.spatial_crop_size or (2, 2)
            crop_stride = self.spatial_crop_stride or crop_size
            crop_h = min(int(crop_size[0]), features.size(2))
            crop_w = min(int(crop_size[1]), features.size(3))
            stride_h = max(1, min(int(crop_stride[0]), crop_h))
            stride_w = max(1, min(int(crop_stride[1]), crop_w))
            windows = features.unfold(2, crop_h, stride_h).unfold(3, crop_w, stride_w)
            pooled = windows.mean(dim=(-1, -2))
            tokens = pooled.permute(0, 2, 3, 1).contiguous()
            return tokens.view(tokens.size(0), -1, tokens.size(-1))
        if self.spatial_feature_mode == "pool_crop":
            pool_size = self.spatial_pool_size or (2, 2)
            pooled = F.adaptive_avg_pool2d(features, pool_size)
            pool_tokens = pooled.flatten(2).transpose(1, 2)
            crop_size = self.spatial_crop_size or (2, 2)
            crop_stride = self.spatial_crop_stride or crop_size
            crop_h = min(int(crop_size[0]), features.size(2))
            crop_w = min(int(crop_size[1]), features.size(3))
            stride_h = max(1, min(int(crop_stride[0]), crop_h))
            stride_w = max(1, min(int(crop_stride[1]), crop_w))
            windows = features.unfold(2, crop_h, stride_h).unfold(3, crop_w, stride_w)
            pooled_windows = windows.mean(dim=(-1, -2))
            crop_tokens = pooled_windows.permute(0, 2, 3, 1).contiguous()
            crop_tokens = crop_tokens.view(crop_tokens.size(0), -1, crop_tokens.size(-1))
            return torch.cat([pool_tokens, crop_tokens], dim=1)

        raise ValueError(f"Unsupported spatial_feature_mode: {self.spatial_feature_mode}")

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

        # DocSP: extract spatial tokens from pre-pixel-shuffle features
        docsp_tokens = None
        if self.use_docsp:
            # [B, H, W, C] → [B, C, H, W] for conv operations
            raw_features = vit_embeds.permute(0, 3, 1, 2).contiguous()
            docsp_tokens = self.docsp(raw_features)  # [B, N_spatial, llm_dim]

        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        extra_tokens = self._extract_spatial_tokens(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        if extra_tokens is not None:
            vit_embeds = torch.cat([vit_embeds, extra_tokens.to(vit_embeds.dtype)], dim=1)
        vit_embeds = self.mlp1(vit_embeds)

        # Concatenate DocSP spatial tokens after mlp1-projected patch tokens
        if docsp_tokens is not None:
            vit_embeds = torch.cat([vit_embeds, docsp_tokens.to(vit_embeds.dtype)], dim=1)

        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
