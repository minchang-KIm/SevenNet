(ase_calculator)=
# ASE Calculator

SevenNet provides an ASE calculator through `sevenn.calculator`.

## Stable calculator classes

- `SevenNetCalculator`: SevenNet energy, force, stress, and atomic energies
- `SevenNetD3Calculator`: `SevenNetCalculator` plus the CUDA D3 correction

## Loading a model

Stable inputs for `SevenNetCalculator`:

- pretrained keyword
- checkpoint path
- in-memory `AtomGraphSequential` model instance
- deployed TorchScript model, when `file_type="torchscript"`

Examples:

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator(model="7net-0")
```

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator(model="path/to/checkpoint_best.pth", file_type="checkpoint")
```

## Multi-modal checkpoints

If the checkpoint carries a modality map, `modal` is required:

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator(model="7net-mf-ompa", modal="mpa")
```

## Accelerator backends

Stable accelerator flags:

- `enable_flash=True`
- `enable_cueq=True`
- `enable_oeq=True`

Use at most one accelerator backend at a time.

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator(model="7net-0", enable_flash=True)
```

Important limits:

- Accelerator selection is supported on checkpoint-backed calculator loading.
- cuEquivariance, OpenEquivariance, and pairgeom are disabled automatically for `file_type="model_instance"` and `file_type="torchscript"`.
- `device="auto"` selects CUDA when available, otherwise CPU.

## Experimental pairgeom

`enable_pairgeom=True` is experimental and only applies to checkpoint-backed calculator use:

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator(
    model="path/to/checkpoint_best.pth",
    file_type="checkpoint",
    enable_pairgeom=True,
)
```

This path is:

- inference-only
- checkpoint-only
- not a deployment mode

For the exact scope and constraints, see {ref}`pairgeom-experimental`.

## D3 calculator

`SevenNetD3Calculator` combines SevenNet with the CUDA D3 implementation:

```python
from sevenn.calculator import SevenNetD3Calculator

calc = SevenNetD3Calculator(model="7net-0", device="cuda")
```

CPU-only D3 is not provided by this implementation.
