#!/usr/bin/env python3
"""Widen a released DM0.5 checkpoint's action interface from 32 to N dims (G1 SONIC: 78).

DM05 projects the noisy action chunk in and the velocity out through two Linear layers sized by
``config.action_dim`` (32 in every released checkpoint):
    model.action_in_proj.weight   [H, 32]   (+ bias [H], unchanged)
    model.action_out_proj.weight  [32, H]   + bias [32]
G1 SONIC acts in 78 dims (motion_token 64 + hands 7+7), so both are widened and config.json gets
``action_dim = N``. OpenDM's training pipeline ALSO pads/TRUNCATES actions to a hard-coded 32
(``PadAction(32)`` in DM05DataConfig.build_dataset) -- locomanilab's entry point
(locomanilab.runtime.dm05.experiment) replaces that with the model's action_dim; widening the checkpoint
alone would silently drop dims 32..77.

Initialisation of the new dims is deliberately NOT zero. With a zero input column the action
expert never sees the noisy value of that dim, so flow matching cannot learn to denoise it and the
dim decodes to "mean + noise" after a short fine-tune -- exactly what happened to XR-1's widened
dims (C09). New input columns / output rows are drawn from N(0, s) with s matched to the RMS of the
existing ones; new output biases start at 0. Every other tensor is copied bit-for-bit and verified.

    python src/locomanilab/runtime/dm05/widen.py --src <DM05 dir> --dst <DM05-sonic78 dir> --action-dim 78
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_W, OUT_W, OUT_B = "model.action_in_proj.weight", "model.action_out_proj.weight", "model.action_out_proj.bias"
# Copied as-is. norm_stats.json is left out on purpose: it holds the RELEASE's robot profiles
# (DOS W1 / Aloha, 14-D); locomanilab computes its own stats per experiment, and a stale file next to the
# weights is the kind of thing an inference path picks up by accident.
SIDE_FILES = ["chat_template.jinja", "generation_config.json", "preprocessor_config.json",
              "processor_config.json", "tokenizer.json", "tokenizer_config.json", "README.md"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--action-dim", type=int, default=78)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vision-attn", choices=["keep", "sdpa"], default="sdpa",
                    help="the release pins vlm_config.attn_implementation.vision_config to "
                         "flash_attention_2, which SigLIP checks at from_pretrained time (before "
                         "OpenDM's --model-config.vision-attn-implementation applies), so without the "
                         "flash_attn package the model cannot even be built. sdpa (PyTorch's fused "
                         "attention, flash kernel on H100) needs nothing extra.")
    a = ap.parse_args()

    cfg = json.loads((a.src / "config.json").read_text())
    old = int(cfg["action_dim"])
    new = a.action_dim
    if new <= old:
        raise SystemExit(f"--action-dim {new} must exceed the checkpoint's {old}")
    a.dst.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(a.src / "model.safetensors", framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    w_in, w_out, b_out = tensors[IN_W], tensors[OUT_W], tensors[OUT_B]
    hidden = w_in.shape[0]
    assert w_in.shape == (hidden, old) and w_out.shape == (old, hidden) and b_out.shape == (old,), \
        (w_in.shape, w_out.shape, b_out.shape)

    g = torch.Generator().manual_seed(a.seed)
    in_col_rms = w_in.float().pow(2).mean(0).sqrt()      # per input column (one per action dim)
    out_row_rms = w_out.float().pow(2).mean(1).sqrt()    # per output row
    s_in, s_out = float(in_col_rms.median()), float(out_row_rms.median())
    add_in = torch.randn(hidden, new - old, generator=g) * s_in
    add_out = torch.randn(new - old, hidden, generator=g) * s_out
    tensors[IN_W] = torch.cat([w_in, add_in.to(w_in.dtype)], dim=1).contiguous()
    tensors[OUT_W] = torch.cat([w_out, add_out.to(w_out.dtype)], dim=0).contiguous()
    tensors[OUT_B] = torch.cat([b_out, torch.zeros(new - old, dtype=b_out.dtype)]).contiguous()

    save_file(tensors, str(a.dst / "model.safetensors"), metadata=meta or {"format": "pt"})
    cfg["action_dim"] = new
    vision_attn_before = None
    attn = (cfg.get("vlm_config") or {}).get("attn_implementation")
    if a.vision_attn != "keep" and isinstance(attn, dict) and "vision_config" in attn:
        vision_attn_before = attn["vision_config"]
        attn["vision_config"] = a.vision_attn
    (a.dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in SIDE_FILES:
        if (a.src / name).exists():
            shutil.copy2(a.src / name, a.dst / name)

    # verify: untouched tensors bit-identical, widened ones keep the original slice
    changed = {IN_W, OUT_W, OUT_B}
    with safe_open(a.src / "model.safetensors", framework="pt") as fs, \
            safe_open(a.dst / "model.safetensors", framework="pt") as fd:
        assert set(fs.keys()) == set(fd.keys())
        for k in fs.keys():
            if k in changed:
                continue
            if not torch.equal(fs.get_tensor(k), fd.get_tensor(k)):
                raise SystemExit(f"tensor {k} changed")
        assert torch.equal(fd.get_tensor(IN_W)[:, :old], w_in)
        assert torch.equal(fd.get_tensor(OUT_W)[:old], w_out)
        assert torch.equal(fd.get_tensor(OUT_B)[:old], b_out)
    report = {
        "src": str(a.src), "dst": str(a.dst), "action_dim": [old, new], "seed": a.seed,
        "in_col_rms_existing": {"min": float(in_col_rms.min()), "median": s_in, "max": float(in_col_rms.max())},
        "out_row_rms_existing": {"min": float(out_row_rms.min()), "median": s_out, "max": float(out_row_rms.max())},
        "init_new": "N(0, s) with s = median existing RMS; new out bias 0",
        "untouched_tensors_bit_identical": len(tensors) - 3,
        "vlm_vision_attn_implementation": [vision_attn_before, a.vision_attn],
    }
    (a.dst / "WIDEN_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
