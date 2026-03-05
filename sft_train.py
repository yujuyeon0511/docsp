import argparse
import glob
import json
import os
import sys
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Ensure the script's directory is on the path so 'model' package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
# Patch torch.load to use weights_only=False for checkpoint resume compatibility
# (transformers Trainer passes weights_only=True but checkpoints contain numpy objects)
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load
import torchvision.transforms as T
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Register custom model classes so AutoConfig/AutoModel can load them
from model.configuration_internvl_chat import InternVLChatConfig
from model.modeling_internvl_chat import InternVLChatModel
from model.conversation import get_conv_template

AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVLChatConfig, InternVLChatModel)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def resolve_image_path(path: str, image_root: Optional[str]) -> str:
    if os.path.isabs(path) or image_root is None:
        return path
    if path.startswith("images/") and "/images/" in image_root:
        base = image_root.split("/images/")[0] + "/images"
        return os.path.join(base, path[len("images/"):])
    return os.path.join(image_root, path)


def replace_image_tokens(
    prompt: str, num_patches_list: List[int], num_image_token: int
) -> str:
    if not num_patches_list:
        return prompt
    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * (num_image_token * num_patches)
            + IMG_END_TOKEN
        )
        if "<image>" not in prompt:
            raise ValueError("Prompt is missing <image> placeholder for provided images.")
        prompt = prompt.replace("<image>", image_tokens, 1)
    return prompt


def build_prompt(
    messages: List[Tuple[str, str]],
    template_name: str,
    system_message: Optional[str],
    num_patches_list: List[int],
    num_image_token: int,
) -> str:
    template = get_conv_template(template_name)
    if system_message:
        template.system_message = system_message
    for role, content in messages:
        template.append_message(role, content)
    prompt = template.get_prompt()
    return replace_image_tokens(prompt, num_patches_list, num_image_token)


def build_labels(
    tokenizer,
    messages: List[Tuple[str, str]],
    template_name: str,
    system_message: Optional[str],
    num_patches_list: List[int],
    num_image_token: int,
) -> Tuple[List[int], List[int]]:
    prompt = build_prompt(
        messages, template_name, system_message, num_patches_list, num_image_token
    )
    input_ids = tokenizer(
        prompt, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    labels = [-100] * len(input_ids)

    for i, (role, content) in enumerate(messages):
        if role != get_conv_template(template_name).roles[1]:
            continue
        prefix_messages = messages[:i]
        before_messages = prefix_messages + [(role, None)]
        after_messages = prefix_messages + [(role, content)]
        prompt_before = build_prompt(
            before_messages, template_name, system_message, num_patches_list, num_image_token
        )
        prompt_after = build_prompt(
            after_messages, template_name, system_message, num_patches_list, num_image_token
        )
        start = len(
            tokenizer(prompt_before, add_special_tokens=False)["input_ids"]
        )
        end = len(tokenizer(prompt_after, add_special_tokens=False)["input_ids"])
        for idx in range(start, min(end, len(labels))):
            labels[idx] = input_ids[idx]

    return input_ids, labels


class SFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        template_name: str,
        num_image_token: int,
        image_root: Optional[str],
        image_size: int,
        max_num_tiles: int,
        max_length: int,
        drop_long_samples: bool,
        max_samples: Optional[int] = None,
    ) -> None:
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            first_decode_error = None
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    first_decode_error = (line_no, exc)
                    break
            if first_decode_error is not None:
                f.seek(0)
                try:
                    data = json.load(f)
                except json.JSONDecodeError as exc:
                    line_no, first_exc = first_decode_error
                    raise ValueError(
                        f"Failed to parse dataset as JSONL (first error at line {line_no}). "
                        "Also failed to parse as JSON. Please provide JSONL or JSON array."
                    ) from exc
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                if not isinstance(data, list):
                    raise ValueError(
                        "JSON dataset must be a list of samples or a dict with a 'data' list."
                    )
                self.samples = data
        # Subsample if max_samples specified
        if max_samples is not None and max_samples > 0 and len(self.samples) > max_samples:
            import random
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.image_root = image_root
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.max_length = max_length
        self.drop_long_samples = drop_long_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_messages(
        self, sample: Dict
    ) -> Tuple[List[Tuple[str, str]], Optional[str], List[str]]:
        def _normalize_content(content) -> str:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                        continue
                    if isinstance(item, dict):
                        item_type = item.get("type") or item.get("modal")
                        if item_type in ("text", "text_input"):
                            parts.append(item.get("text") or item.get("content") or "")
                        elif item_type in ("image", "image_url", "image_input"):
                            parts.append("<image>")
                        elif "text" in item:
                            parts.append(item.get("text") or "")
                        else:
                            parts.append(str(item))
                    else:
                        parts.append(str(item))
                return "".join(parts)
            return str(content)

        def _collect_image_paths(content) -> List[str]:
            paths: List[str] = []
            if not isinstance(content, list):
                return paths
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type") or item.get("modal")
                if item_type not in ("image", "image_url", "image_input"):
                    continue
                path = item.get("path") or item.get("image") or item.get("url")
                if path:
                    paths.append(path)
            return paths

        system_message = _normalize_content(sample.get("system"))
        raw_messages = sample.get("messages", [])
        raw_conversations = sample.get("conversations", [])
        if not raw_messages and not raw_conversations:
            raise ValueError(
                "Sample must include a non-empty 'messages' list or 'conversations' list."
            )
        template = get_conv_template(self.template_name)
        mapped = []
        image_paths: List[str] = []
        if raw_messages:
            for msg in raw_messages:
                role = msg["role"].lower()
                content = _normalize_content(msg.get("content"))
                image_paths.extend(_collect_image_paths(msg.get("content")))
                if role == "system":
                    system_message = content
                    continue
                if role == "user":
                    mapped.append((template.roles[0], content))
                elif role == "assistant":
                    mapped.append((template.roles[1], content))
                else:
                    raise ValueError(f"Unsupported role: {msg['role']}")
        else:
            for msg in raw_conversations:
                role = msg["from"].lower()
                content = _normalize_content(msg.get("value"))
                image_paths.extend(_collect_image_paths(msg.get("value")))
                if role in ("system",):
                    system_message = content
                    continue
                if role in ("human", "user"):
                    mapped.append((template.roles[0], content))
                elif role in ("gpt", "assistant"):
                    mapped.append((template.roles[1], content))
                else:
                    raise ValueError(f"Unsupported role: {msg['from']}")
        return mapped, system_message, image_paths

    def __getitem__(self, idx: int) -> Dict:
        try:
            return self._getitem_inner(idx)
        except Exception as e:
            print(f"[Dataset] Error at idx {idx}: {e}, skipping to next sample")
            return self.__getitem__((idx + 1) % len(self.samples))

    def _getitem_inner(self, idx: int) -> Dict:
        sample = self.samples[idx]
        messages, system_message, extracted_images = self._extract_messages(sample)

        images = []
        if "images" in sample:
            images = sample["images"]
        elif "image" in sample:
            images = [sample["image"]]
        if extracted_images:
            images = list(images) + extracted_images

        num_patches_list: List[int] = []
        pixel_values_list: List[torch.Tensor] = []

        if images:
            for image_path in images:
                resolved_path = resolve_image_path(image_path, self.image_root)
                tiles = load_image(
                    resolved_path, input_size=self.image_size, max_num=self.max_num_tiles
                )
                num_patches_list.append(tiles.size(0))
                pixel_values_list.append(tiles)

            if "<image>" not in " ".join([m[1] for m in messages]):
                first_role, first_content = messages[0]
                messages[0] = (first_role, "<image>\n" + first_content)

        input_ids, labels = build_labels(
            self.tokenizer,
            messages,
            self.template_name,
            system_message,
            num_patches_list,
            self.num_image_token,
        )

        if len(input_ids) > self.max_length:
            if self.drop_long_samples:
                return self.__getitem__((idx + 1) % len(self.samples))
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    image_size: int
    pixel_dtype: torch.dtype

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            ids = f["input_ids"]
            lbl = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(
                torch.tensor(ids + [self.tokenizer.pad_token_id] * pad_len, dtype=torch.long)
            )
            labels.append(torch.tensor(lbl + [-100] * pad_len, dtype=torch.long))
            attention_mask.append(
                torch.tensor([1] * len(ids) + [0] * pad_len, dtype=torch.long)
            )

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)

        pixel_values_list = [f["pixel_values"] for f in features if f["pixel_values"] is not None]
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=self.pixel_dtype)
            image_flags = torch.ones((pixel_values.size(0), 1), dtype=torch.long)
        else:
            pixel_values = torch.zeros(
                (1, 3, self.image_size, self.image_size), dtype=self.pixel_dtype
            )
            image_flags = torch.zeros((1, 1), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL3.5-8B SFT training")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/johns4378/j_son/InternVL3_5-8B",
    )
    parser.add_argument("--train_jsonl", type=str, nargs="+", default=None,
                        help="One or more JSONL/JSON training data files")
    parser.add_argument("--datasets_conf", type=str, default=None,
                        help="Tab-separated config file: JSONL_PATH<TAB>IMAGE_ROOT[<TAB>MAX_SAMPLES]")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_root", type=str, nargs="+", default=None,
                        help="One or more image root dirs (paired with train_jsonl, or single for all)")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_num_tiles", type=int, default=12)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--drop_long_samples", action="store_true")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument(
        "--optim",
        type=str,
        default="adafactor",
        choices=["adamw_torch", "adamw_hf", "adamw_torch_fused", "adamw_apex_fused", "adafactor"],
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--llm_use_pretrained", action="store_true")
    parser.add_argument("--llm_model_name_or_path", type=str, default=None)
    parser.add_argument("--llm_trust_remote_code", action="store_true")
    parser.add_argument("--ignore_mismatched_sizes", action="store_true")
    parser.add_argument("--train_projector_only", action="store_true")
    parser.add_argument("--freeze_vision", action="store_true",
                        help="Freeze the vision encoder (train LLM + projector only). Standard for Stage 2 SFT.")
    parser.add_argument("--keep_projector_weights", action="store_true",
                        help="When used with --llm_use_pretrained, load mlp1 from model_path instead of reinitializing")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON file (e.g. ds_zero2.json)")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for the language model")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume training, or 'auto' to find latest in output_dir")
    parser.add_argument("--use_docsp", action="store_true",
                        help="Enable DocSP spatial perception projector")
    return parser.parse_args()


def load_filtered_state_dict(
    model_path: str, skip_prefixes: Tuple[str, ...]
) -> dict:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(model_path, shard_file)
        shard_state = load_safetensors(shard_path)
        for key, value in shard_state.items():
            if key.startswith(skip_prefixes):
                continue
            state_dict[key] = value
    return state_dict


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    # Load config using our registered InternVLChatConfig (not trust_remote_code
    # which would load the original code from model_path without DocSP support)
    config = InternVLChatConfig.from_pretrained(args.model_path)
    if args.use_docsp:
        config.use_docsp = True
    if args.llm_use_pretrained:
        if not args.llm_model_name_or_path:
            raise ValueError("--llm_use_pretrained requires --llm_model_name_or_path")
        config.llm_use_pretrained = True
        config.llm_model_name_or_path = args.llm_model_name_or_path
        config.llm_trust_remote_code = args.llm_trust_remote_code or args.llm_use_pretrained

    state_dict = None
    if args.llm_use_pretrained:
        if args.keep_projector_weights:
            # Stage 2: keep trained mlp1 + docsp from model_path, only skip LLM weights
            skip_prefixes = ("language_model.",)
        else:
            # Stage 1: skip LLM and projectors (will be reinitialized)
            skip_prefixes = ("language_model.", "mlp1.", "docsp.")
        state_dict = load_filtered_state_dict(
            args.model_path, skip_prefixes=skip_prefixes
        )

    ignore_mismatched_sizes = args.ignore_mismatched_sizes or args.llm_use_pretrained or args.use_docsp
    if state_dict is None:
        model = InternVLChatModel.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=args.use_flash_attn,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
    else:
        model = InternVLChatModel(config, use_flash_attn=args.use_flash_attn)
        model = model.to(dtype=model_dtype)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: missing keys when loading state_dict: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Warning: unexpected keys when loading state_dict: {len(unexpected_keys)}")

    tokenizer_path = args.llm_model_name_or_path if args.llm_use_pretrained else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )
    if args.llm_use_pretrained:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<img>", "</img>", "<IMG_CONTEXT>"]}
        )
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    template_name = model.config.template

    if args.train_projector_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.mlp1.parameters():
            param.requires_grad = True
        if hasattr(model, 'docsp') and model.use_docsp:
            for param in model.docsp.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Training projector only: trainable params {trainable}/{total}")
    elif args.freeze_vision:
        for param in model.vision_model.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Freeze vision: trainable params {trainable}/{total}")

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        # Freeze everything first, then LoRA on LLM + projector trainable
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.mlp1.parameters():
            param.requires_grad = True
        if hasattr(model, 'docsp') and model.use_docsp:
            for param in model.docsp.parameters():
                param.requires_grad = True
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA training: trainable params {trainable}/{total}")
        model.language_model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Build dataset entries: list of (jsonl_path, image_root, max_samples)
    dataset_entries = []
    if args.datasets_conf:
        with open(args.datasets_conf, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                jsonl_path = parts[0]
                image_root = parts[1] if len(parts) > 1 and parts[1] != "__NONE__" else None
                max_samples = int(parts[2]) if len(parts) > 2 and parts[2] else None
                dataset_entries.append((jsonl_path, image_root, max_samples))
    elif args.train_jsonl:
        jsonl_list = args.train_jsonl
        if args.image_root is None:
            root_list = [None] * len(jsonl_list)
        else:
            sanitized = [None if r == "__NONE__" else r for r in args.image_root]
            if len(sanitized) == 1:
                root_list = sanitized * len(jsonl_list)
            elif len(sanitized) == len(jsonl_list):
                root_list = sanitized
            else:
                raise ValueError(
                    f"--image_root must be 1 value (shared) or match --train_jsonl count "
                    f"({len(jsonl_list)}), got {len(args.image_root)}"
                )
        for jsonl_path, image_root in zip(jsonl_list, root_list):
            dataset_entries.append((jsonl_path, image_root, None))
    else:
        raise ValueError("Either --train_jsonl or --datasets_conf must be specified")

    datasets = []
    for jsonl_path, image_root, max_samples in dataset_entries:
        ds = SFTDataset(
            jsonl_path=jsonl_path,
            tokenizer=tokenizer,
            template_name=template_name,
            num_image_token=model.num_image_token,
            image_root=image_root,
            image_size=args.image_size,
            max_num_tiles=args.max_num_tiles,
            max_length=args.max_length,
            drop_long_samples=args.drop_long_samples,
            max_samples=max_samples,
        )
        print(f"Loaded {len(ds)} samples from {jsonl_path} (image_root={image_root}, max_samples={max_samples})")
        datasets.append(ds)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)
        print(f"Total combined dataset: {len(dataset)} samples")

    data_collator = DataCollator(
        tokenizer=tokenizer, image_size=args.image_size, pixel_dtype=model_dtype
    )

    # Auto-adjust batch size based on GPU memory
    per_device_batch = args.per_device_train_batch_size
    grad_accum = args.gradient_accumulation_steps
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if gpu_mem_gb < 50 and per_device_batch > 1:
            scale = per_device_batch
            per_device_batch = 1
            grad_accum = grad_accum * scale
            print(f"[Auto-GPU] {gpu_mem_gb:.0f}GB GPU detected → "
                  f"batch_size={per_device_batch}, grad_accum={grad_accum}")
        else:
            print(f"[Auto-GPU] {gpu_mem_gb:.0f}GB GPU detected → "
                  f"batch_size={per_device_batch}, grad_accum={grad_accum} (default)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        remove_unused_columns=False,
        report_to=[],
        deepspeed=args.deepspeed,
        ddp_timeout=7200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    # Resume from checkpoint if specified
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "auto":
        # Find latest checkpoint in output_dir
        import re
        ckpts = [d for d in os.listdir(args.output_dir)
                 if os.path.isdir(os.path.join(args.output_dir, d)) and re.match(r"checkpoint-\d+", d)]
        if ckpts:
            resume_ckpt = os.path.join(args.output_dir, max(ckpts, key=lambda x: int(x.split("-")[-1])))
            print(f"Auto-resuming from {resume_ckpt}")
        else:
            resume_ckpt = None
            print("No checkpoint found, starting from scratch")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    if args.use_lora:
        # Merge LoRA weights and save full model
        model.language_model = model.language_model.merge_and_unload()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Copy custom model code files so the saved model is self-contained
    model_code_files = [
        "configuration_internvl_chat.py",
        "modeling_internvl_chat.py",
        "configuration_intern_vit.py",
        "modeling_intern_vit.py",
        "conversation.py",
        "docsp_projector.py",
    ]
    # Look in model/ subdir first (DocSP layout), then model_path root (original layout)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_code_dir = os.path.join(script_dir, "model")
    for fname in model_code_files:
        src = os.path.join(model_code_dir, fname)
        if not os.path.isfile(src):
            src = os.path.join(args.model_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))


if __name__ == "__main__":
    main()
