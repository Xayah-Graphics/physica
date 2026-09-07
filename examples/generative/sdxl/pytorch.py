# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "torch==2.14.0+cu132", "triton-windows==3.8.0.post28",
#   "tokenizers==0.23.2", "ftfy==6.3.1", "safetensors==0.8.0",
#   "numpy==2.5.3", "pillow==12.3.0", "huggingface-hub==1.30.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu132" }
# [[tool.uv.index]]
# name = "pytorch-cu132"
# url = "https://download.pytorch.org/whl/cu132"
# explicit = true
# ///
"""Independent SDXL inference reference for oneObsession_v22.

Run in the prepared project environment:
    .venv/Scripts/python.exe examples/generative/sdxl/pytorch.py \
        <checkpoint.safetensors> <output-directory> [--compile]

On Windows, launch --compile from the x64 Visual Studio developer environment.
Inductor uses 30 compilation workers; compilation errors propagate directly.

Edit the configuration below to change the image. All model forwards are native
PyTorch; the tokenizer vocabulary is the pinned, official OpenAI CLIP asset.
CLIP computes in FP32 from the checkpoint's FP16 values; UNet computes in FP16,
VAE in BF16, and the Euler state and update remain FP32.
Checkpoint module names describe the original SDXL weight layout. Inference-only
Sequential blocks omit dropout, with the corresponding keys remapped at loading.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import html
import importlib.metadata
import json
import math
import re
import sys
import time

import ftfy
from huggingface_hub import hf_hub_download
import numpy
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
import torch
from torch import nn
from torch.nn import functional


@dataclass
class Configuration:
    positive: str = "petite, realistic, photorealistic, black hair, long hair, \nblue serafuku, red neckerchief, blue sailor collar, long sleeves, pleated skirt, blue beret, \nskinny, black thighhighs, slim legs,\n1girl, indoors, solo, portrait, looking at viewer, \n\n(anime CG, realistic painting style), masterpiece, best quality, amazing quality, newest, very aesthetic,newest, highres, year 2025, high resolution, excellent, medium resolution, clean coloring，soft shading, "
    negative: str = "see-through thighhighs, red rope,\n\nworst quality,normal quality,quality,lowres,anatomical nonsense,bad anatomy,bad hands, mutated hands,interlocked fingers,extra fingers,watermark,low resolution, old,transparent,low logo, text, username, signature, early, watermark, signature, jpeg artifacts, username, censored, lowres, logo, text, sketch, multiple views, monochrome, thick thighs, "
    seed: int = 659915289870415
    width: int = 1024
    height: int = 1536
    steps: int = 50
    cfg: float = 4.5


CONFIGURATION = Configuration()
TOKENIZER_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
LATENT_SCALE = 0.13025


def weighted_segments(text: str, weight: float = 1.0) -> list[tuple[str, float]]:
    """Parse balanced parentheses, absolute :weights, and escaped parentheses."""
    segments = []
    depth = 0
    begin = 0
    for match in re.finditer(r"\\[()]|[()]", text):
        symbol = match.group()
        if symbol.startswith("\\"):
            continue
        if symbol == "(":
            if depth == 0:
                if match.start() > begin:
                    segments.append((text[begin:match.start()], weight))
                begin = match.end()
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                inner = text[begin:match.start()]
                suffix = re.search(r":([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$", inner)
                inner_weight = weight * 1.1
                if suffix is not None and suffix.start() > 0:
                    inner_weight = float(suffix[1])
                    inner = inner[:suffix.start()]
                segments.extend(weighted_segments(inner, inner_weight))
                begin = match.end()
    if begin < len(text):
        segments.append((text[begin:], weight))
    return segments


class PromptTokenizer:
    def __init__(self) -> None:
        self.path = Path(hf_hub_download("openai/clip-vit-large-patch14", "tokenizer.json", revision=TOKENIZER_REVISION))
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def encode(self, text: str, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunks = []
        chunk = [(49406, 1.0)]
        for segment, weight in weighted_segments(text):
            segment = segment.replace(r"\(", "(").replace(r"\)", ")")
            segment = html.unescape(html.unescape(ftfy.fix_text(segment)))
            segment = re.sub(r"\s+", " ", segment).strip()
            group = self.tokenizer.encode(segment, add_special_tokens=False).ids
            split_group = len(group) >= 8
            offset = 0
            while offset < len(group):
                space = 76 - len(chunk)
                if len(group) - offset <= space:
                    chunk.extend((token, weight) for token in group[offset:])
                    offset = len(group)
                else:
                    if split_group:
                        chunk.extend((token, weight) for token in group[offset:offset + space])
                        offset += space
                    chunk.append((49407, 1.0))
                    chunk.extend([(pad, 1.0)] * (77 - len(chunk)))
                    chunks.append(chunk)
                    chunk = [(49406, 1.0)]
        chunk.append((49407, 1.0))
        chunk.extend([(pad, 1.0)] * (77 - len(chunk)))
        chunks.append(chunk)
        tokens = torch.tensor([[token for token, _ in row] for row in chunks], dtype=torch.int64)
        weights = torch.tensor([[value for _, value in row] for row in chunks], dtype=torch.float32)
        return tokens, weights


class ClipBlock(nn.Module):
    def __init__(self, width: int, heads: int, quick_gelu: bool) -> None:
        super().__init__()
        self.heads = heads
        self.quick_gelu = quick_gelu
        self.norm1 = nn.LayerNorm(width, eps=1.0e-5)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, eps=1.0e-5)
        self.expand = nn.Linear(width, 4 * width)
        self.contract = nn.Linear(4 * width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = values.shape
        qkv = self.qkv(self.norm1(values)).reshape(batch, sequence, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attention = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        values = values + self.projection(attention.transpose(1, 2).reshape(batch, sequence, width))
        hidden = self.expand(self.norm2(values))
        hidden = hidden * torch.sigmoid(1.702 * hidden) if self.quick_gelu else functional.gelu(hidden)
        return values + self.contract(hidden)


class ClipTextEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, quick_gelu: bool) -> None:
        super().__init__()
        self.token = nn.Embedding(49408, width)
        self.position = nn.Parameter(torch.empty(77, width))
        self.blocks = nn.ModuleList(ClipBlock(width, heads, quick_gelu) for _ in range(layers))
        self.norm = nn.LayerNorm(width, eps=1.0e-5)
        self.projection = nn.Linear(width, width, bias=False) if width == 1280 else nn.Identity()

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.token(tokens) + self.position
        for index, block in enumerate(self.blocks):
            values = block(values)
            if index == len(self.blocks) - 2:
                penultimate = values
        end = (tokens == 49407).to(torch.int64).argmax(dim=-1)
        pooled = self.norm(values)[torch.arange(tokens.shape[0], device=tokens.device), end]
        return penultimate.float(), self.projection(pooled).float()

    def encode(self, tokens: torch.Tensor, weights: torch.Tensor, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
        weighted = bool((weights != 1.0).any())
        count = tokens.shape[0]
        if weighted:
            empty = torch.tensor([[49406, 49407] + [pad] * 75], dtype=torch.int64)
            tokens = torch.cat((tokens, empty))
        sequence, pooled = self(tokens.to(self.position.device))
        if weighted:
            weights = weights.to(sequence.device)[..., None]
            sequence = torch.where(weights != 1.0, sequence[-1:] + weights * (sequence[:count] - sequence[-1:]), sequence[:count])
        return sequence.reshape(1, count * 77, -1), pooled[:1]


def timestep_embedding(times: torch.Tensor, width: int) -> torch.Tensor:
    frequency = torch.exp(-math.log(10000.0) * torch.arange(width // 2, dtype=torch.float32, device=times.device) / (width // 2))
    angles = times.float().reshape(-1, 1) * frequency
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class ResBlock(nn.Module):
    def __init__(self, input_width: int, width: int) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(nn.GroupNorm(32, input_width), nn.SiLU(), nn.Conv2d(input_width, width, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(1280, width))
        self.out_layers = nn.Sequential(nn.GroupNorm(32, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))
        self.skip_connection = nn.Conv2d(input_width, width, 1) if input_width != width else nn.Identity()

    def forward(self, values: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_layers(values) + self.emb_layers(embedding)[:, :, None, None]
        return self.skip_connection(values) + self.out_layers(hidden)


class Attention(nn.Module):
    def __init__(self, width: int, context_width: int) -> None:
        super().__init__()
        self.heads = width // 64
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(context_width, width, bias=False)
        self.to_v = nn.Linear(context_width, width, bias=False)
        self.to_out = nn.Linear(width, width)

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = values.shape
        query = self.to_q(values).reshape(batch, sequence, self.heads, 64).transpose(1, 2)
        key = self.to_k(context).reshape(batch, -1, self.heads, 64).transpose(1, 2)
        value = self.to_v(context).reshape(batch, -1, self.heads, 64).transpose(1, 2)
        attention = functional.scaled_dot_product_attention(query, key, value)
        return self.to_out(attention.transpose(1, 2).reshape(batch, sequence, width))


class GEGLU(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, 8 * width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(values).chunk(2, dim=-1)
        return value * functional.gelu(gate)


class TransformerBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn1 = Attention(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.attn2 = Attention(width, 2048)
        self.norm3 = nn.LayerNorm(width)
        self.ff = nn.Sequential(GEGLU(width), nn.Linear(4 * width, width))

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(values)
        values = values + self.attn1(normalized, normalized)
        values = values + self.attn2(self.norm2(values), context)
        return values + self.ff(self.norm3(values))


class SpatialTransformer(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, width, eps=1.0e-6)
        self.proj_in = nn.Linear(width, width)
        self.transformer_blocks = nn.ModuleList(TransformerBlock(width) for _ in range(depth))
        self.proj_out = nn.Linear(width, width)

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, width, height, columns = values.shape
        hidden = self.norm(values).flatten(2).transpose(1, 2)
        hidden = self.proj_in(hidden)
        for block in self.transformer_blocks:
            hidden = block(hidden, context)
        hidden = self.proj_out(hidden).transpose(1, 2).reshape(batch, width, height, columns)
        return values + hidden


class Upsample(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(functional.interpolate(values, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(320, 1280), nn.SiLU(), nn.Linear(1280, 1280))
        self.label_emb = nn.Sequential(nn.Linear(2816, 1280), nn.SiLU(), nn.Linear(1280, 1280))
        self.input_blocks = nn.ModuleList([nn.ModuleList([nn.Conv2d(4, 320, 3, padding=1)])])
        width = 320
        skips = [width]
        for level, (next_width, depth) in enumerate(zip((320, 640, 1280), (0, 2, 10))):
            for _ in range(2):
                block = nn.ModuleList([ResBlock(width, next_width)])
                width = next_width
                if depth:
                    block.append(SpatialTransformer(width, depth))
                self.input_blocks.append(block)
                skips.append(width)
            if level < 2:
                self.input_blocks.append(nn.ModuleList([nn.Conv2d(width, width, 3, stride=2, padding=1)]))
                skips.append(width)
        self.middle_block = nn.ModuleList([ResBlock(1280, 1280), SpatialTransformer(1280, 10), ResBlock(1280, 1280)])
        self.output_blocks = nn.ModuleList()
        for level, (next_width, depth) in enumerate(zip((1280, 640, 320), (10, 2, 0))):
            for index in range(3):
                block = nn.ModuleList([ResBlock(width + skips.pop(), next_width)])
                width = next_width
                if depth:
                    block.append(SpatialTransformer(width, depth))
                if index == 2 and level < 2:
                    block.append(Upsample(width))
                self.output_blocks.append(block)
        self.out = nn.Sequential(nn.GroupNorm(32, 320), nn.SiLU(), nn.Conv2d(320, 4, 3, padding=1))

    def forward(self, values: torch.Tensor, times: torch.Tensor, context: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        embedding = self.time_embed(timestep_embedding(times, 320).to(values.dtype)) + self.label_emb(condition)
        skips = []
        for block in self.input_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    values = layer(values, embedding)
                elif isinstance(layer, SpatialTransformer):
                    values = layer(values, context)
                else:
                    values = layer(values)
            skips.append(values)
        values = self.middle_block[0](values, embedding)
        values = self.middle_block[1](values, context)
        values = self.middle_block[2](values, embedding)
        for block in self.output_blocks:
            values = torch.cat((values, skips.pop()), dim=1)
            for layer in block:
                if isinstance(layer, ResBlock):
                    values = layer(values, embedding)
                elif isinstance(layer, SpatialTransformer):
                    values = layer(values, context)
                else:
                    values = layer(values)
        return self.out(values)


class VaeResBlock(nn.Module):
    def __init__(self, input_width: int, width: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, input_width, eps=1.0e-6)
        self.conv1 = nn.Conv2d(input_width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, width, eps=1.0e-6)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(input_width, width, 1) if input_width != width else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(functional.silu(self.norm1(values)))
        hidden = self.conv2(functional.silu(self.norm2(hidden)))
        return self.nin_shortcut(values) + hidden


class VaeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, 512, eps=1.0e-6)
        self.q = nn.Conv2d(512, 512, 1)
        self.k = nn.Conv2d(512, 512, 1)
        self.v = nn.Conv2d(512, 512, 1)
        self.proj_out = nn.Conv2d(512, 512, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(values)
        query = self.q(normalized).flatten(2).transpose(1, 2).unsqueeze(1)
        key = self.k(normalized).flatten(2).transpose(1, 2).unsqueeze(1)
        value = self.v(normalized).flatten(2).transpose(1, 2).unsqueeze(1)
        hidden = functional.scaled_dot_product_attention(query, key, value)
        return values + self.proj_out(hidden.squeeze(1).transpose(1, 2).reshape_as(values))


class VaeLevel(nn.Module):
    def __init__(self, input_width: int, width: int, upsample: bool) -> None:
        super().__init__()
        self.block = nn.Sequential(VaeResBlock(input_width, width), VaeResBlock(width, width), VaeResBlock(width, width))
        self.upsample = Upsample(width) if upsample else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.block(values))


class VaeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        self.mid = nn.Sequential(VaeResBlock(512, 512), VaeAttention(), VaeResBlock(512, 512))
        self.up = nn.ModuleList([VaeLevel(256, 128, False), VaeLevel(512, 256, True), VaeLevel(512, 512, True), VaeLevel(512, 512, True)])
        self.norm_out = nn.GroupNorm(32, 128, eps=1.0e-6)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        values = self.conv_in(self.post_quant_conv(latent))
        values = self.mid(values)
        for level in reversed(self.up):
            values = level(values)
        return self.conv_out(functional.silu(self.norm_out(values)))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        values = self((latent / LATENT_SCALE).to(torch.bfloat16))
        return (values.float() * 0.5 + 0.5).clamp(0.0, 1.0)


def load_models(checkpoint: Path) -> tuple[ClipTextEncoder, ClipTextEncoder, UNet, VaeDecoder]:
    with torch.device("meta"):
        clip_l = ClipTextEncoder(768, 12, 12, True)
        clip_g = ClipTextEncoder(1280, 32, 20, False)
        unet = UNet()
        vae = VaeDecoder()
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        for model, large in ((clip_l, False), (clip_g, True)):
            root = "conditioner.embedders.1.model." if large else "conditioner.embedders.0.transformer.text_model."
            mapping = {"token.weight": root + ("token_embedding.weight" if large else "embeddings.token_embedding.weight"),
                       "position": root + ("positional_embedding" if large else "embeddings.position_embedding.weight"),
                       "norm.weight": root + ("ln_final.weight" if large else "final_layer_norm.weight"),
                       "norm.bias": root + ("ln_final.bias" if large else "final_layer_norm.bias")}
            for index in range(len(model.blocks)):
                prefix = root + (f"transformer.resblocks.{index}." if large else f"encoder.layers.{index}.")
                for target, original in (("norm1", "ln_1" if large else "layer_norm1"),
                                         ("norm2", "ln_2" if large else "layer_norm2"),
                                         ("projection", "attn.out_proj" if large else "self_attn.out_proj"),
                                         ("expand", "mlp.c_fc" if large else "mlp.fc1"),
                                         ("contract", "mlp.c_proj" if large else "mlp.fc2")):
                    for suffix in ("weight", "bias"):
                        mapping[f"blocks.{index}.{target}.{suffix}"] = prefix + original + "." + suffix
            state = {target: source.get_tensor(original) for target, original in mapping.items()}
            for index in range(len(model.blocks)):
                for suffix in ("weight", "bias"):
                    if large:
                        qkv = source.get_tensor(root + f"transformer.resblocks.{index}.attn.in_proj_{suffix}")
                    else:
                        qkv = torch.cat([source.get_tensor(root + f"encoder.layers.{index}.self_attn.{component}_proj.{suffix}") for component in ("q", "k", "v")])
                    state[f"blocks.{index}.qkv.{suffix}"] = qkv
            if large:
                state["projection.weight"] = source.get_tensor(root + "text_projection").T.contiguous()
            model.load_state_dict(state, assign=True)
            model.to(device="cuda", dtype=torch.float32).eval().requires_grad_(False)
        state = {}
        for key in source.keys():
            if key.startswith("model.diffusion_model."):
                target = key.removeprefix("model.diffusion_model.")
                target = target.replace(".out_layers.3.", ".out_layers.2.").replace(".to_out.0.", ".to_out.")
                target = target.replace(".ff.net.0.", ".ff.0.").replace(".ff.net.2.", ".ff.1.")
                target = target.replace("label_emb.0.", "label_emb.")
                target = target.replace("input_blocks.3.0.op.", "input_blocks.3.0.").replace("input_blocks.6.0.op.", "input_blocks.6.0.")
                state[target] = source.get_tensor(key)
        unet.load_state_dict(state, assign=True)
        unet.to(device="cuda", dtype=torch.float16).eval().requires_grad_(False)
        state = {}
        for key in source.keys():
            if key.startswith("first_stage_model.decoder.") or key.startswith("first_stage_model.post_quant_conv."):
                target = key.removeprefix("first_stage_model.").removeprefix("decoder.")
                target = target.replace("mid.block_1.", "mid.0.").replace("mid.attn_1.", "mid.1.").replace("mid.block_2.", "mid.2.")
                state[target] = source.get_tensor(key)
        vae.load_state_dict(state, assign=True)
        vae.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    return clip_l, clip_g, unet, vae


def encode_conditioning(tokenizer: PromptTokenizer, clip_l: ClipTextEncoder, clip_g: ClipTextEncoder, configuration: Configuration) -> tuple[torch.Tensor, torch.Tensor]:
    contexts = []
    pooled = []
    for text in (configuration.positive, configuration.negative):
        tokens_l, weights_l = tokenizer.encode(text, 49407)
        tokens_g, weights_g = tokenizer.encode(text, 0)
        sequence_l, _ = clip_l.encode(tokens_l, weights_l, 49407)
        sequence_g, pooled_g = clip_g.encode(tokens_g, weights_g, 0)
        contexts.append(torch.cat((sequence_l, sequence_g), dim=-1))
        pooled.append(pooled_g)
    sequence = math.lcm(contexts[0].shape[1], contexts[1].shape[1])
    context = torch.cat([value.repeat(1, sequence // value.shape[1], 1) for value in contexts])
    sizes = torch.tensor([configuration.height, configuration.width, 0, 0, configuration.height, configuration.width], device="cuda")
    geometry = timestep_embedding(sizes, 256).reshape(1, 1536).repeat(2, 1)
    condition = torch.cat((torch.cat(pooled), geometry), dim=1)
    return context.to(torch.float16), condition.to(torch.float16)


def noise_schedule(steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(math.sqrt(0.00085), math.sqrt(0.012), 1000, dtype=torch.float64).square()
    alpha = (1.0 - betas).cumprod(dim=0)
    training_sigmas = ((1.0 - alpha) / alpha).sqrt().float()
    times = torch.tensor([999 - int(index * 1000 / steps) for index in range(steps)], dtype=torch.int64)
    sigmas = torch.cat((training_sigmas[times], torch.zeros(1)))
    return sigmas.to("cuda"), times.to(device="cuda", dtype=torch.float32)


def sample(unet: nn.Module, context: torch.Tensor, condition: torch.Tensor, configuration: Configuration) -> torch.Tensor:
    sigmas, times = noise_schedule(configuration.steps)
    generator = torch.Generator(device="cpu").manual_seed(configuration.seed)
    noise = torch.randn((1, 4, configuration.height // 8, configuration.width // 8), generator=generator, dtype=torch.float32)
    state = noise.to("cuda") * (1.0 + sigmas[0].square()).sqrt()
    begin = time.perf_counter()
    for step in range(configuration.steps):
        torch.compiler.cudagraph_mark_step_begin()
        sigma = sigmas[step]
        model_input = (state / (1.0 + sigma.square()).sqrt()).repeat(2, 1, 1, 1).to(torch.float16)
        epsilon = unet(model_input, times[step].expand(2), context, condition).float()
        positive, negative = (state - sigma * epsilon).chunk(2)
        denoised = negative + configuration.cfg * (positive - negative)
        derivative = (state - denoised) / sigma
        state = state + derivative * (sigmas[step + 1] - sigma)
        if (step + 1) % 5 == 0 or step + 1 == configuration.steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            print(f"SAMPLE {step + 1:3d}/{configuration.steps}  {(step + 1) / elapsed:.2f} steps/s  ETA {elapsed * (configuration.steps - step - 1) / (step + 1):.1f}s", flush=True)
    return state


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compile", action="store_true", dest="compile_unet")
    arguments = parser.parse_args()
    configuration = CONFIGURATION
    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.cuda.reset_peak_memory_stats()
    timing = {}
    total_begin = time.perf_counter()
    print(f"START Physica / SDXL / PyTorch {torch.__version__}\nGPU {torch.cuda.get_device_name()}\nMODE {'compiled' if arguments.compile_unet else 'eager'}\nCHECKPOINT {arguments.checkpoint}", flush=True)
    begin = time.perf_counter()
    tokenizer = PromptTokenizer()
    clip_l, clip_g, unet, vae = load_models(arguments.checkpoint)
    torch.cuda.synchronize()
    timing["load_seconds"] = time.perf_counter() - begin
    print(f"LOAD {timing['load_seconds']:.2f}s", flush=True)
    begin = time.perf_counter()
    context, condition = encode_conditioning(tokenizer, clip_l, clip_g, configuration)
    torch.cuda.synchronize()
    timing["encode_seconds"] = time.perf_counter() - begin
    print(f"ENCODE {timing['encode_seconds']:.2f}s  context={tuple(context.shape)}", flush=True)
    timing["compile_seconds"] = 0.0
    warm_input = torch.zeros((2, 4, configuration.height // 8, configuration.width // 8), dtype=torch.float16, device="cuda")
    warm_time = torch.tensor(999.0, device="cuda").expand(2)
    if arguments.compile_unet:
        from torch._inductor import config as inductor_config

        inductor_config.compile_threads = 30
        inductor_config.worker_start_method = "spawn"
        begin = time.perf_counter()
        unet = torch.compile(unet, fullgraph=True, dynamic=False, mode="max-autotune")
        print("COMPILE UNet", flush=True)
        unet(warm_input, warm_time, context, condition)
        torch.cuda.synchronize()
        timing["compile_seconds"] = time.perf_counter() - begin
    begin = time.perf_counter()
    torch.compiler.cudagraph_mark_step_begin()
    unet(warm_input, warm_time, context, condition)
    torch.cuda.synchronize()
    timing["warmup_seconds"] = time.perf_counter() - begin
    del warm_input, warm_time
    print(f"READY compile={timing['compile_seconds']:.2f}s  warmup={timing['warmup_seconds']:.2f}s", flush=True)
    begin = time.perf_counter()
    latent = sample(unet, context, condition, configuration)
    torch.cuda.synchronize()
    timing["sample_seconds"] = time.perf_counter() - begin
    begin = time.perf_counter()
    pixels = vae.decode(latent)
    torch.cuda.synchronize()
    timing["decode_seconds"] = time.perf_counter() - begin
    print(f"DECODE {timing['decode_seconds']:.2f}s", flush=True)
    begin = time.perf_counter()
    pixels = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0.0, 255.0).astype(numpy.uint8)
    Image.fromarray(pixels).save(arguments.output / "image.png")
    save_file({"latent": latent.cpu().contiguous()}, arguments.output / "latent.safetensors", metadata={"format": "sdxl-scaled-latent", "scaling_factor": str(LATENT_SCALE)})
    timing["write_seconds"] = time.perf_counter() - begin
    timing["total_seconds"] = time.perf_counter() - total_begin
    metadata = {
        "configuration": asdict(configuration), "checkpoint": str(arguments.checkpoint.resolve()),
        "sampler": "euler", "scheduler": "simple", "denoise": 1.0, "batch": 1,
        "mode": "compiled" if arguments.compile_unet else "eager", "tokenizer_revision": TOKENIZER_REVISION,
        "precision": {"clip_weights_source": "float16", "clip_compute": "float32", "unet": "float16", "vae": "bfloat16", "sampling": "float32"},
        "python": sys.version, "python_executable": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("triton-windows", "tokenizers", "ftfy", "safetensors", "numpy", "pillow", "huggingface-hub")},
        "seconds": timing, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (arguments.output / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WRITE {arguments.output.resolve()}\nDONE {timing['total_seconds']:.2f}s  peak {metadata['peak_allocated_bytes'] / 1024 ** 3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
