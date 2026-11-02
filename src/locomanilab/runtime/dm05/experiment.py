#!/usr/bin/env python3
"""locomanilab's DM0.5 (OpenDM) experiment entry: SFT on G1 SONIC data + inference runtime.

Run through OpenDM's launcher exactly like its own playground entries:
    script/dm05_launcher.sh --exp <locomanilab>/src/locomanilab/runtime/dm05/experiment.py --task train \
        --nproc_per_node 4 --data-config.dataset-name locomanilab_train50_merged \
        --model-config.model-name-or-path <DM05-sonic78> --trainer-config.output-dir <run dir> ...

Differences from upstream ``playground/dm05_sft_demo.py`` (the reference SFT recipe, whose
optimiser / schedule / precision defaults are kept: AdamW 2.5e-5, 1000 warmup, cosine to 10 %,
fp32-mixed, per-device batch 8, full fine-tune):

1. ACTION WIDTH. ``DM05DataConfig.build_dataset`` ends with a hard-coded ``PadAction(32)``,
   which pads -- or TRUNCATES -- every action to 32 dims. G1 SONIC's action is 78-D
   (motion_token 64 + hands 7+7) on a checkpoint widened to ``action_dim = 78``
   (locomanilab.runtime.dm05.widen), so the upstream pipeline would silently drop dims 32..77
   (both hands included). The pad width is taken from the model's config.json instead, and
   training refuses to start if the dataset's action width differs from it.
2. ABSOLUTE actions. Relative mode subtracts the state from the action and needs
   action_dim == state_dim; SONIC's latent motion_token (78) vs joint state (43) cannot be
   differenced. The converter writes an explicit 78-D ``action`` per frame.
3. One ego camera ("Head"), robot_type "Unitree G1", 43-D joint state rendered as state tokens.
4. Datasets are registered from conversion records via OPENDM_DATA_PATH (set below, before
   anything imports ``opendm.dataset.register``), not by editing the OpenDM checkout.
6. Vision attention defaults to sdpa (upstream: flash_attention_2, which needs the flash_attn
   package); the widened base's config.json pins the same (widen_action_dim.py --vision-attn).
5. Norm stats default to ``<output_dir>/norm_stats`` (upstream: CWD-relative ``./norm_stats``,
   shared by every run started from the same directory).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

os.environ.setdefault("OPENDM_DATA_PATH", str(Path(__file__).resolve().parent / "registry"))
os.environ.setdefault("WANDB_MODE", "disabled")

import tyro  # noqa: E402

import opendm.exp.dm05_exp as _upstream  # noqa: E402
from opendm.constants.precision import FP32_MIXED_PRECISION_POLICY  # noqa: E402
from opendm.constants.robot import ActionMode  # noqa: E402
from opendm.data.transforms import PadAction  # noqa: E402

_PAD_WIDTH: list[int] = []


def _pad_action_to_model_width(shared_dim: int = 32) -> PadAction:
    """Stand-in for upstream's ``PadAction(32)``: the literal is ignored on purpose."""
    del shared_dim
    if not _PAD_WIDTH:
        raise RuntimeError("action pad width not set -- build the dataset through LocoManiLabDM05Exp")
    return PadAction(_PAD_WIDTH[0])


def _model_action_dim(model_path: str | None) -> int:
    if not model_path:
        raise ValueError("set model_config.model_name_or_path to a local prepared checkpoint")
    return int(json.loads((Path(model_path) / "config.json").read_text())["action_dim"])


@dataclass
class LocoManiLabDM05DataConfig(_upstream.DM05DataConfig):
    dataset_name: str = field(default="locomanilab_train50_merged")
    action_mode: ActionMode = field(default=ActionMode.ABSOLUTE)
    norm_stats_root: str = field(default="")  # "" -> <trainer output_dir>/norm_stats


@dataclass
class LocoManiLabDM05ModelConfig(_upstream.DM05ModelConfig):
    model_name_or_path: str | None = field(default=None)
    precision_policy: Literal["bf16_mixed", "fp32_mixed"] = field(default=FP32_MIXED_PRECISION_POLICY)
    # upstream default flash_attention_2 needs the flash_attn package (no wheel for torch 2.11/cu130);
    # sdpa runs PyTorch's fused (flash) kernel on H100 and needs nothing extra. Train AND serve.
    vision_attn_implementation: Literal["auto", "eager", "sdpa", "flash_attention_2"] = field(default="sdpa")


@dataclass
class LocoManiLabDM05OptimizerConfig(_upstream.DM05OptimizerConfig):
    base_lr: float = field(default=2.5e-5)


@dataclass
class LocoManiLabDM05TrainerConfig(_upstream.DM05TrainerConfig):
    output_dir: str = field(default="user_checkpoints/locomanilab_dm05")
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=3)
    num_train_steps: int = field(default=50000)
    save_only_model: bool = field(default=False)
    logging_steps: int = field(default=10)


@dataclass
class LocoManiLabDM05InferenceConfig(_upstream.DM05InferenceConfig):
    output_action_dim: int = field(default=78)
    image_prompts: list[str] = field(default_factory=lambda: ["Head"])


@dataclass
class LocoManiLabDM05Exp(_upstream.DM05Exp):
    use_lora: bool | None = field(default=False)
    model_config: LocoManiLabDM05ModelConfig = field(default_factory=LocoManiLabDM05ModelConfig)
    optimizer_config: LocoManiLabDM05OptimizerConfig = field(default_factory=LocoManiLabDM05OptimizerConfig)
    trainer_config: LocoManiLabDM05TrainerConfig = field(default_factory=LocoManiLabDM05TrainerConfig)
    data_config: LocoManiLabDM05DataConfig = field(default_factory=LocoManiLabDM05DataConfig)
    inference_config: LocoManiLabDM05InferenceConfig = field(default_factory=LocoManiLabDM05InferenceConfig)

    def __post_init__(self):
        super().__post_init__()
        if not self.data_config.norm_stats_root:
            self.data_config.norm_stats_root = str(Path(self.trainer_config.output_dir) / "norm_stats")
        width = _model_action_dim(self.model_config.model_name_or_path)
        _PAD_WIDTH[:] = [width]
        # build_dataset resolves the name PadAction from the upstream module's globals
        _upstream.PadAction = _pad_action_to_model_width
        if self.inference_config.output_action_dim != width and self.task == "inference":
            raise SystemExit(f"inference output_action_dim {self.inference_config.output_action_dim} "
                             f"!= checkpoint action_dim {width}")

    def _initialize_train(self):
        super()._initialize_train()
        width = _PAD_WIDTH[0]
        sample = self.trainer.train_dataset[0]
        got = tuple(sample["action"].shape)
        mask_dims = int(sample["action_mask"].reshape(-1, got[-1]).any(0).sum())
        if got[-1] != width or mask_dims != width:
            raise SystemExit(f"dataset action {got} (mask covers {mask_dims} dims) does not match the "
                             f"checkpoint's action_dim {width}")
        _upstream.logger.info(f"[locomanilab] action chunk {got}, all {mask_dims} dims supervised")


if __name__ == "__main__":
    exp = tyro.cli(LocoManiLabDM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")
