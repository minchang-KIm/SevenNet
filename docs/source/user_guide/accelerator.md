# Accelerator Installation

This page covers installation and support boundaries for the optional tensor-product accelerators used by SevenNet. For the behavioral model and correctness boundaries, see {ref}`accelerators`. For the experimental `pairgeom` feature, see {ref}`pairgeom-experimental`.

Stable support matrix:

| Backend | Training | Checkpoint inference / ASE | LAMMPS Torch export | LAMMPS ML-IAP export |
| --- | --- | --- | --- | --- |
| FlashTP | Yes | Yes | Yes | Yes |
| cuEquivariance | Yes | Yes | No | Yes |
| OpenEquivariance | Yes | Yes | Yes | Yes |

Use at most one tensor-product accelerator at a time in stable workflows.

## cuEquivariance

Requirements:

- Python >= 3.10
- A CUDA-capable GPU
- `cuequivariance`, `cuequivariance-torch`, and the matching CUDA ops package

Install via the packaged extras:

```bash
pip install sevenn[cueq12]  # CUDA 12.x
pip install sevenn[cueq13]  # CUDA 13.x
```

Availability check:

```bash
python -c "from sevenn.nn.cue_helper import is_cue_available; print(is_cue_available())"
```

## FlashTP

Requirements:

- Python >= 3.10
- A CUDA-capable GPU
- flashTP built for the target GPU architecture

Install from the FlashTP source tree:

```bash
git clone https://github.com/SNU-ARC/flashTP.git
cd flashTP
pip install -r requirements.txt
CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
```

Availability check:

```bash
python -c "from sevenn.nn.flash_helper import is_flash_available; print(is_flash_available())"
```

## OpenEquivariance

Requirements:

- Python >= 3.10
- A CUDA-capable GPU
- `openequivariance`

Install via the packaged extra:

```bash
pip install sevenn[oeq]
```

Availability check:

```bash
python -c "from sevenn.nn.oeq_helper import is_oeq_available; print(is_oeq_available())"
```

## Stable usage surfaces

Training:

```bash
sevenn train input.yaml --enable_flash
sevenn train input.yaml --enable_cueq
sevenn train input.yaml --enable_oeq
```

Inference:

```bash
sevenn inference checkpoint_best.pth structures/*.extxyz --enable_flash
sevenn inference checkpoint_best.pth structures/*.extxyz --enable_cueq
sevenn inference checkpoint_best.pth structures/*.extxyz --enable_oeq
```

Experimental pairgeom + FlashTP inference:

```bash
sevenn inference checkpoint_best.pth structures/*.extxyz --enable_flash --enable_pairgeom --pairgeom_backend flash
```

ASE calculator:

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator("7net-0", enable_flash=True)
```

Experimental checkpoint-backed ASE calculator with pairgeom + FlashTP:

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator(
    "path/to/checkpoint_best.pth",
    file_type="checkpoint",
    enable_flash=True,
    enable_pairgeom=True,
    pairgeom_backend="flash",
)
```

LAMMPS export:

- TorchScript export supports FlashTP and OpenEquivariance.
- ML-IAP export supports FlashTP, cuEquivariance, and OpenEquivariance.

`pairgeom` is not an accelerator backend. It is an experimental inference-only reuse path with separate constraints. When `pairgeom_backend="flash"` is selected successfully, SevenNet still uses FlashTP's directed-edge kernel path; pairgeom only changes the pair-invariant preprocessing that feeds that path.
