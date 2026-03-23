# Pretrained Checkpoints

This page documents the pretrained checkpoint keywords recognized by the current SevenNet codebase. It intentionally avoids ranking, benchmark, or recommendation claims.

## Stable keyword surface

The current keyword resolver accepts the following canonical pretrained names:

- `7net-0`
- `7net-0_22may2024`
- `7net-l3i5`
- `7net-mf-0`
- `7net-mf-ompa`
- `7net-omat`
- `7net-omni`
- `7net-omni-i8`
- `7net-omni-i12`

`sevennet-*` spellings are also accepted where applicable.

These keywords can be used anywhere SevenNet accepts a checkpoint specifier:

- `SevenNetCalculator`
- `sevenn inference`
- `sevenn get_model`
- checkpoint continuation in `input.yaml`

If the checkpoint file is not already present in the package directory or cache, SevenNet attempts to resolve it through the configured release download link.

## Inspect before use

Use `sevenn cp <model>` to inspect a pretrained checkpoint before selecting it for inference or deployment:

```bash
sevenn cp 7net-0
sevenn cp 7net-omni
```

This is the stable way to confirm:

- cutoff
- interaction depth
- parity setting
- whether a modal/task map is present
- whether the checkpoint was saved with a tensor-product accelerator backend enabled

## Modal checkpoints

Some pretrained checkpoints are multi-modal and require `modal` or `--modal` during inference or deployment.

Stable examples:

- `7net-mf-0`
- `7net-mf-ompa`
- `7net-omni`
- `7net-omni-i8`
- `7net-omni-i12`

Modal names are part of the checkpoint metadata. Do not guess them. Inspect the checkpoint first:

```bash
sevenn cp 7net-mf-ompa
sevenn cp 7net-omni
```

## Usage examples

ASE calculator:

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator("7net-0")
```

Checkpoint inference:

```bash
sevenn inference 7net-0 structures/*.extxyz
```

Checkpoint inference for a modal checkpoint:

```bash
sevenn inference 7net-mf-ompa structures/*.extxyz --modal mpa
```

LAMMPS export:

```bash
sevenn get_model 7net-0
sevenn get_model 7net-mf-ompa --modal mpa
```

## Notes on scope

- Pretrained keyword resolution is a checkpoint-loading convenience, not a separate model API.
- Accelerator flags such as `--enable_flash`, `--enable_cueq`, and `--enable_oeq` select runtime/export backends when supported; they do not change the checkpoint architecture.
- Experimental `pairgeom` is not a pretrained-model feature. It is an inference-time execution mode for checkpoint-backed inference only.
