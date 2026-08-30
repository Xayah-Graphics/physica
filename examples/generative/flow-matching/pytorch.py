"""Independent FP32 reference for the CUDA-native FlowDiT implementation.

Usage:
    python examples/generative/flow-matching/pytorch.py \
        data/cifar-10-batches-bin <output-directory> 20000

Verified with Python 3.14.2, PyTorch 2.10.0+cu130, NumPy 2.3.5,
Pillow 12.1.0, and Safetensors 0.7.0.
"""

from pathlib import Path
import csv
import math
import sys
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from safetensors.torch import save_file


WIDTH = 256
SEQUENCE = 256
HEADS = 8
HEAD_WIDTH = 32
MLP_WIDTH = 1024
BLOCKS = 8
BATCH = 256
EMA_HALF_LIFE_SAMPLES = 500_000
EMA_RAMP_UP_RATIO = 0.05
MODEL_SEED = 42
TRAINING_SEED = 1234
SAMPLING_SEED = 42


def patchify(images: torch.Tensor) -> torch.Tensor:
    batch = images.shape[0]
    return images.permute(0, 2, 3, 1).reshape(batch, 16, 2, 16, 2, 3).permute(0, 1, 3, 2, 4, 5).reshape(batch, SEQUENCE, 12)


def unpatchify(patches: torch.Tensor) -> torch.Tensor:
    batch = patches.shape[0]
    return patches.reshape(batch, 16, 16, 2, 2, 3).permute(0, 5, 1, 3, 2, 4).reshape(batch, 3, 32, 32)


def position_embedding() -> torch.Tensor:
    frequency_index = torch.arange(WIDTH // 4, dtype=torch.float32)
    frequency = torch.exp(-math.log(10_000.0) * frequency_index / (WIDTH // 4))
    y = torch.arange(16, dtype=torch.float32).repeat_interleave(16)
    x = torch.arange(16, dtype=torch.float32).repeat(16)
    return torch.cat((
        torch.sin(y[:, None] * frequency),
        torch.cos(y[:, None] * frequency),
        torch.sin(x[:, None] * frequency),
        torch.cos(x[:, None] * frequency),
    ), dim=1)


def time_embedding(times: torch.Tensor) -> torch.Tensor:
    frequency_index = torch.arange(WIDTH // 2, device=times.device, dtype=torch.float32)
    frequency = torch.exp(-math.log(10_000.0) * frequency_index / (WIDTH // 2))
    angle = times[:, None] * frequency
    return torch.cat((torch.cos(angle), torch.sin(angle)), dim=1)


def modulate(values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return values * (1.0 + scale[:, None, :]) + shift[:, None, :]


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.output = nn.Linear(WIDTH, WIDTH)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = values.shape
        qkv = self.qkv(values).reshape(batch, sequence, 3, HEADS, HEAD_WIDTH).permute(2, 0, 3, 1, 4)
        attention = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.output(attention.transpose(1, 2).reshape(batch, sequence, WIDTH))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_normalization = nn.LayerNorm(WIDTH, eps=1.0e-6, elementwise_affine=False)
        self.attention = Attention()
        self.mlp_normalization = nn.LayerNorm(WIDTH, eps=1.0e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, MLP_WIDTH), nn.GELU(approximate="tanh"), nn.Linear(MLP_WIDTH, WIDTH))
        self.modulation = nn.Linear(WIDTH, 6 * WIDTH)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        attention_shift, attention_scale, attention_gate, mlp_shift, mlp_scale, mlp_gate = self.modulation(functional.silu(condition)).chunk(6, dim=1)
        values = values + attention_gate[:, None, :] * self.attention(modulate(self.attention_normalization(values), attention_shift, attention_scale))
        values = values + mlp_gate[:, None, :] * self.mlp(modulate(self.mlp_normalization(values), mlp_shift, mlp_scale))
        return values


class FlowDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Linear(12, WIDTH)
        self.register_buffer("position", position_embedding())
        self.classes = nn.Embedding(11, WIDTH)
        self.time = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.blocks = nn.ModuleList(Block() for _ in range(BLOCKS))
        self.final_normalization = nn.LayerNorm(WIDTH, eps=1.0e-6, elementwise_affine=False)
        self.final_modulation = nn.Linear(WIDTH, 2 * WIDTH)
        self.velocity = nn.Linear(WIDTH, 12)
        self.initialize()

    def initialize(self) -> None:
        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(initialize_linear)
        nn.init.normal_(self.classes.weight, std=0.02)
        nn.init.normal_(self.time[0].weight, std=0.02)
        nn.init.normal_(self.time[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation.weight)
            nn.init.zeros_(block.modulation.bias)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        nn.init.zeros_(self.velocity.weight)
        nn.init.zeros_(self.velocity.bias)

    def forward(self, patches: torch.Tensor, times: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        values = self.patch(patches) + self.position
        condition = self.time(time_embedding(times)) + self.classes(labels)
        for block in self.blocks:
            values = block(values, condition)
        shift, scale = self.final_modulation(functional.silu(condition)).chunk(2, dim=1)
        return self.velocity(modulate(self.final_normalization(values), shift, scale))


def load_cifar10(directory: Path) -> tuple[torch.Tensor, torch.Tensor]:
    records = []
    for index in range(1, 6):
        records.append(numpy.fromfile(directory / f"data_batch_{index}.bin", dtype=numpy.uint8).reshape(-1, 3073))
    records = numpy.concatenate(records, axis=0)
    labels = torch.from_numpy(records[:, 0].copy()).to(torch.int64)
    images = torch.from_numpy(records[:, 1:].copy()).reshape(-1, 3, 32, 32)
    return images, labels


def sample(model: FlowDiT, path: Path, device: torch.device) -> None:
    generator = torch.Generator(device=device).manual_seed(SAMPLING_SEED)
    state = patchify(torch.randn((100, 3, 32, 32), generator=generator, device=device))
    labels = torch.arange(10, device=device).repeat_interleave(10)

    def velocity(values: torch.Tensor, sample_time: float) -> torch.Tensor:
        combined = torch.cat((values, values), dim=0)
        times = torch.full((200,), sample_time, device=device)
        combined_labels = torch.cat((labels, torch.full_like(labels, 10)))
        conditional, unconditional = model(combined, times, combined_labels).chunk(2, dim=0)
        return unconditional + 2.0 * (conditional - unconditional)

    with torch.no_grad():
        for step in range(25):
            first_time = step / 25.0
            second_time = (step + 1) / 25.0
            first_velocity = velocity(state, first_time)
            predictor = state + (second_time - first_time) * first_velocity
            second_velocity = velocity(predictor, second_time)
            state += 0.5 * (second_time - first_time) * (first_velocity + second_velocity)

    pixels = unpatchify(state).clamp(-1.0, 1.0)
    pixels = ((pixels + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    grid = numpy.zeros((320, 320, 3), dtype=numpy.uint8)
    for index in range(100):
        row = index // 10
        column = index % 10
        grid[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32] = pixels[index]
    Image.fromarray(grid).save(path)


def save_ema(parameter_names: list[str], ema: list[torch.Tensor], path: Path, step: int) -> None:
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in zip(parameter_names, ema)},
        path,
        metadata={
            "step": str(step),
            "precision": "fp32-ieee",
            "ema_half_life_samples": str(EMA_HALF_LIFE_SAMPLES),
            "ema_ramp_up_ratio": str(EMA_RAMP_UP_RATIO),
            "system": "pytorch-flow-dit-reference",
        },
    )


def save_milestone(model: FlowDiT, parameters: list[nn.Parameter], ema: list[torch.Tensor], samples_directory: Path, step: int, device: torch.device) -> None:
    raw = [parameter.detach().clone() for parameter in parameters]
    sample(model, samples_directory / f"step-{step:06d}-parameters.png", device)
    with torch.no_grad():
        torch._foreach_copy_(parameters, ema)
    sample(model, samples_directory / f"step-{step:06d}-ema.png", device)
    with torch.no_grad():
        torch._foreach_copy_(parameters, raw)


dataset_directory = Path(sys.argv[1])
output_directory = Path(sys.argv[2])
end_step = int(sys.argv[3])
samples_directory = output_directory / "samples"
checkpoints_directory = output_directory / "checkpoints"
samples_directory.mkdir(parents=True, exist_ok=True)
checkpoints_directory.mkdir(parents=True, exist_ok=True)

torch.manual_seed(MODEL_SEED)
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
device = torch.device("cuda")
images, dataset_labels = load_cifar10(dataset_directory)
images = images.to(device)
dataset_labels = dataset_labels.to(device)
model = FlowDiT().to(device)
parameters = list(model.parameters())
parameter_names = [name for name, _ in model.named_parameters()]
optimizer = torch.optim.AdamW(parameters, lr=1.0e-4, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
ema = [parameter.detach().clone() for parameter in parameters]
generator = torch.Generator(device=device).manual_seed(TRAINING_SEED)

print(
    "Physica / Flow Matching / PyTorch Reference\n\n"
    f"  dataset    {dataset_directory}\n"
    f"  device     {device}\n"
    f"  precision  FP32 IEEE\n"
    f"  target     {end_step} steps\n"
    f"  batch      {BATCH}\n"
    f"  seed       model={MODEL_SEED}, training={TRAINING_SEED}, sampling={SAMPLING_SEED}\n"
    f"  torch      {torch.__version__}, CUDA {torch.version.cuda}\n"
    f"  output     {output_directory}\n",
    flush=True,
)

csv_path = output_directory / "training.csv"
with csv_path.open("w", newline="") as file:
    csv.writer(file).writerow(("step", "loss", "samples_per_second", "elapsed_seconds", "ema_half_life_samples", "ema_decay", "gradient_norm", "parameter_max"))

training_elapsed = 0.0
interval_begin = time.perf_counter()
loss_sum = torch.zeros((), device=device)
for step in range(1, end_step + 1):
    indices = torch.randint(50_000, (BATCH,), generator=generator, device=device)
    data = images[indices].to(torch.float32) * (2.0 / 255.0) - 1.0
    flips = torch.rand((BATCH,), generator=generator, device=device) < 0.5
    data[flips] = torch.flip(data[flips], dims=(3,))
    labels = dataset_labels[indices]
    labels = torch.where(torch.rand((BATCH,), generator=generator, device=device) < 0.1, 10, labels)
    noise = torch.randn((BATCH, 3, 32, 32), generator=generator, device=device)
    times = torch.rand((BATCH,), generator=generator, device=device)
    data = patchify(data)
    noise = patchify(noise)
    path = noise + times[:, None, None] * (data - noise)
    target = data - noise

    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(path, times, labels) - target) ** 2)
    loss.backward()
    for block in model.blocks:
        block.attention.qkv.bias.grad[WIDTH:2 * WIDTH].zero_()
    if step % 100 == 0:
        gradient_norm = torch.sqrt(sum(torch.sum(parameter.grad.square()) for parameter in parameters)).item()
    optimizer.step()

    processed_samples = (step - 1) * BATCH
    if processed_samples == 0:
        ema_half_life_samples = 0.0
        ema_decay = 0.0
    else:
        ema_half_life_samples = min(float(EMA_HALF_LIFE_SAMPLES), processed_samples * EMA_RAMP_UP_RATIO)
        ema_decay = 2.0 ** (-BATCH / ema_half_life_samples)
    with torch.no_grad():
        torch._foreach_lerp_(ema, parameters, 1.0 - ema_decay)
    loss_sum += loss.detach()

    if step % 100 == 0:
        average_loss = (loss_sum / 100.0).item()
        parameter_max = max(parameter.detach().abs().max().item() for parameter in parameters)
        torch.cuda.synchronize()
        training_elapsed += time.perf_counter() - interval_begin
        samples_per_second = step * BATCH / training_elapsed
        eta = (end_step - step) * training_elapsed / step
        print(f"{step:7d} / {end_step:7d}  loss {average_loss:.6f}  grad {gradient_norm:.4f}  parameter {parameter_max:.4f}  {samples_per_second:7.1f} samples/s  ema {ema_decay:.6f}  eta {eta / 60.0:5.1f} min", flush=True)
        with csv_path.open("a", newline="") as file:
            csv.writer(file).writerow((step, average_loss, samples_per_second, training_elapsed, ema_half_life_samples, ema_decay, gradient_norm, parameter_max))
        loss_sum.zero_()
        interval_begin = time.perf_counter()

    if step % 1_000 == 0:
        milestone_begin = time.perf_counter()
        save_milestone(model, parameters, ema, samples_directory, step, device)
        if step % 5_000 == 0:
            save_ema(parameter_names, ema, checkpoints_directory / f"step-{step:06d}-ema.safetensors", step)
        print(f"saved raw + EMA milestone {step} in {time.perf_counter() - milestone_begin:.1f}s", flush=True)
        interval_begin = time.perf_counter()

save_ema(parameter_names, ema, output_directory / "final-ema.safetensors", end_step)
print(f"complete in {training_elapsed / 60.0:.1f} training minutes", flush=True)
