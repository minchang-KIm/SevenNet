# Technical Reference

This page documents the current stable user-facing behavior of SevenNet for preprocessing, training, inference, and deployment. Experimental features are isolated in Section 7.

# 1. Overview

SevenNet is an equivariant graph neural network interatomic potential framework. Its core model is a NequIP-style message-passing architecture built from spherical-harmonic edge features, learned radial filters, equivariant tensor products, gated interaction blocks, and energy-based force/stress evaluation.

Stable runtime concepts:

- Topology: neighbor connectivity and periodic-image bookkeeping, primarily `edge_index` and `pbc_shift`.
- Geometry: position-dependent edge vectors and derived quantities such as distances, radial basis values, cutoff values, and spherical harmonics.
- Computation: learned per-layer operations such as filter generation, tensor products, message aggregation, node updates, and readout.

Core computation pipeline:

1. Build or load a graph for the current snapshot.
2. Form edge geometry from the current snapshot geometry for that graph representation.
3. Encode each edge with radial basis, cutoff, and spherical harmonics.
4. Run NequIP-style interaction blocks with learned tensor-product filters.
5. Read out atomic energies and sum them to total energy.
6. Obtain forces and stress by automatic differentiation.

Important invariants:

- `edge_index` is topology, not geometry.
- If positions change, geometry-dependent quantities must be recomputed for the new snapshot.
- FlashTP, cuEquivariance, and OpenEquivariance are kernel/backend changes, not architecture changes.

# 2. CLI Interface

SevenNet has five workflow commands, one checkpoint utility command, and one deployment helper:

- Main workflow commands: `preset`, `graph_build`, `train`, `inference`, `get_model`
- Utility command: `cp` (alias of `checkpoint`)
- Deployment helper: `patch_lammps`

This is the source of the common “five vs six” confusion: `patch_lammps` is a real subcommand, but it is an environment-preparation helper for LAMMPS source trees rather than part of the core preprocess/train/infer/deploy model workflow.

Stable command behavior:

| Command | Purpose | Input | Output | When to use |
| --- | --- | --- | --- | --- |
| `sevenn preset` | Print a maintained training template | Preset name | YAML to stdout | Start a new training or fine-tuning configuration |
| `sevenn graph_build` | Preprocess raw structures into a `sevenn_data/*.pt` dataset | ASE-readable files or `structure_list`, plus cutoff | `sevenn_data/{name}.pt` and `sevenn_data/{name}.yaml` | Reuse a fixed snapshot dataset across training or inference |
| `sevenn train` or `sevenn` | Train a checkpointed model from `input.yaml` | `input.yaml` | Logs, learning-curve CSV, checkpoints, and optionally generated datasets in the working directory | Main training entry point |
| `sevenn inference` | Run checkpoint inference on structures or graph datasets | Checkpoint or pretrained name plus targets | `errors.txt`, `info.csv`, `per_graph.csv`, `per_atom.csv`, and optionally `sevenn_data/saved_graph.pt` | Evaluate a model without deployment |
| `sevenn get_model` | Export a deployment artifact for LAMMPS | Checkpoint or pretrained name | TorchScript artifact(s) or an ML-IAP wrapper `.pt` | Deploy a trained model |
| `sevenn cp` | Inspect a checkpoint or derive YAML from it | Checkpoint or pretrained name | Summary to stdout, YAML to stdout, or `checkpoint_modal_appended.pth` for modal append | Inspect, reproduce, or continue from checkpoints |

`sevenn` defaults to `train` when no subcommand is given, so `sevenn input.yaml` and `sevenn train input.yaml` are equivalent.

(sevenn-preset)=
## `sevenn preset`

Current preset names:

- `base`
- `fine_tune`
- `fine_tune_le`
- `multi_modal`
- `mf_ompa_fine_tune`
- `sevennet-0`
- `sevennet-l3i5`

Use `sevenn preset <name> > input.yaml` to create a starting configuration. The command writes YAML to stdout; it does not create files by itself.

(sevenn-graph-build)=
## `sevenn graph_build`

`sevenn graph_build <source> <cutoff>` converts raw structures into a stable on-disk graph dataset under `OUT/sevenn_data/`.

Stable outputs:

- `OUT/sevenn_data/<filename>.pt`: collated `SevenNetGraphDataset`
- `OUT/sevenn_data/<filename>.yaml`: metadata and dataset statistics

Notes:

- The standard output format is the `.pt` plus sibling `.yaml` pair.
- `--legacy` keeps the deprecated `.sevenn_data` path for backward compatibility only.
- If you move a processed dataset, move the full `sevenn_data` directory without renaming its contents.

(sevenn-train)=
## `sevenn train`

`sevenn train input.yaml` runs the default `train_v2` training pipeline. `train_v1` remains available for backward compatibility.

Stable training flags:

- `--enable_cueq`
- `--enable_flash`
- `--enable_oeq`
- `--distributed`
- `--distributed_backend {nccl,mpi}`

Use at most one tensor-product accelerator at a time in stable workflows.

(sevenn-inference)=
## `sevenn inference`

`sevenn inference <checkpoint> <targets...>` evaluates a checkpoint or pretrained model on:

- ASE-readable structure files
- `structure_list`
- existing `sevenn_data/*.pt` datasets

Key options:

- `--device {auto,cpu,cuda,cuda:N}`
- `--batch`
- `--save_graph`
- `--allow_unlabeled`
- `--modal`
- `--enable_cueq`
- `--enable_flash`
- `--enable_oeq`
- `--enable_pairgeom` (experimental; see Section 7.1)

`--save_graph` and `--allow_unlabeled` are mutually exclusive.

(sevenn-get-model)=
## `sevenn get_model`

`sevenn get_model` exports a deployment artifact from a checkpoint or pretrained name. `deploy` is an alias of `get_model`.

Stable export modes:

- TorchScript serial deployment: default, writes `deployed_serial.pt`
- TorchScript parallel deployment: `--get_parallel`, writes a directory of `deployed_parallel_<layer>.pt`
- LAMMPS ML-IAP deployment: `--use_mliap`, writes `deployed_serial_mliap.pt`

Stable accelerator compatibility:

- TorchScript deployment: `--enable_flash` and `--enable_oeq`
- ML-IAP deployment: `--enable_cueq`, `--enable_flash`, or `--enable_oeq`
- `--enable_cueq` is not supported for the TorchScript LAMMPS interface
- `--use_mliap --get_parallel` is rejected by the CLI

`get_model` is for deployment export. It is not the normal inference path for checkpoint evaluation.

(sevenn-cp)=
## `sevenn cp`

`sevenn cp` is the checkpoint utility. `checkpoint` is the canonical subcommand name and `cp` is its alias.

Stable functions:

- Print a checkpoint summary
- Print YAML derived from a checkpoint with `--get_yaml {reproduce,continue,continue_modal}`
- Append modal metadata with `--append_modal_yaml`, which writes `checkpoint_modal_appended.pth`

This command is the supported way to inspect checkpoint metadata from the CLI.

## `sevenn patch_lammps`

`sevenn patch_lammps <lammps_dir>` patches a LAMMPS source tree with the SevenNet pair-style sources before compilation.

Stable flags:

- `--d3`
- `--enable_flash`
- `--enable_oeq`

This command edits the LAMMPS source tree in place. It does not deploy a model by itself.

# 3. Data Pipeline

SevenNet uses snapshot graphs. A graph is built for a fixed atomic configuration and stores both topology and that snapshot’s geometry.

Stable `.pt` dataset layout:

- Location: `ROOT/sevenn_data/<name>.pt`
- Required sibling metadata: `ROOT/sevenn_data/<name>.yaml`
- Container type: `SevenNetGraphDataset` backed by PyG `InMemoryDataset`

Typical per-graph fields in the standard path:

| Field | Meaning | Category |
| --- | --- | --- |
| `atomic_numbers` | Atomic numbers per atom | Node data |
| `pos` | Atomic positions of the stored snapshot | Geometry input |
| `edge_index` | Directed neighbor topology | Topology |
| `edge_vec` | Directed edge vectors for the stored snapshot | Geometry |
| `total_energy`, `force_of_atoms`, `stress` | Reference labels when available | Labels |
| `cell_volume`, `num_atoms` | Graph-level metadata | Metadata |
| `data_info` | Structure metadata safe for batching | Metadata |

Optional fields used only when needed:

| Field | Meaning |
| --- | --- |
| `cell_lattice_vectors`, `pbc_shift` | Required when runtime geometry must be reconstructed from positions and periodic images |
| `pair_index`, `pair_shift`, `edge_to_pair`, `edge_is_reversed`, `pair_owner` | Pair-level metadata used by experimental pairgeom |

`graph_build` performs offline snapshot preprocessing:

1. Read raw structures through ASE or `structure_list`.
2. Build the directed neighbor list at the requested cutoff.
3. Store topology and the current snapshot geometry in graph form.
4. Compute dataset statistics and write them to the sibling YAML file.

What is precomputed in the stable dataset path:

- Directed neighbor topology for the stored snapshot
- Stored snapshot edge vectors
- Labels and metadata
- Dataset statistics in `<name>.yaml`

What is not precomputed by `graph_build`:

- Radial basis values
- Cutoff values
- Spherical harmonics
- Per-layer weight-network outputs
- Tensor products
- Message aggregation

Critical distinction:

- A stored `.pt` dataset contains geometry for a fixed snapshot.
- If positions change at runtime, `edge_vec`, distances, and all derived geometry must be recomputed for the new snapshot.
- `edge_index` alone is never sufficient to recover geometry.

# 4. Training

The default training path is `train_v2`.

Stable training pipeline:

1. Read `input.yaml` and validate the `model`, `train`, and `data` sections.
2. Resolve preset values, checkpoint continuation settings, and optional accelerator flags.
3. Load an existing `sevenn_data/*.pt` dataset in place, or preprocess raw input files into `WORKDIR/sevenn_data/`.
4. Compute or load dataset statistics used for `shift`, `scale`, and `conv_denominator` initialization when those fields are specified symbolically.
5. Build the NequIP-style equivariant model.
6. Wrap the model in the trainer, optimizer, scheduler, and error recorder.
7. Run epoch training, validation, CSV logging, and checkpoint writing.

Stable checkpoint outputs from `train_v2`:

- `checkpoint_0.pth` before training updates
- `checkpoint_best.pth` when the selected validation metric improves
- `checkpoint_<epoch>.pth` every `per_epoch`
- `lc.csv` on rank 0
- `log.sevenn` in the working directory

DDP behavior:

- SevenNet uses PyTorch DistributedDataParallel.
- `nccl` uses `LOCAL_RANK`, `RANK`, and `WORLD_SIZE`, and is the standard GPU path.
- `mpi` uses OpenMPI environment variables.
- The configured `batch_size` is process-local. Under DDP, the effective global batch size is `batch_size × world_size`.
- The training loader uses `DistributedSampler`.
- When `train_shuffle` is enabled, the sampler epoch is advanced each epoch with `set_epoch(epoch)`.
- Rank 0 writes CSV logs and checkpoints.

Accelerator flags during training:

- `--enable_flash`: patch the convolution path to FlashTP kernels
- `--enable_cueq`: patch the convolution path to cuEquivariance kernels
- `--enable_oeq`: patch the convolution path to OpenEquivariance kernels

These flags change kernel/backend execution, not the model architecture or graph semantics.

# 5. Inference

Stable inference execution flow:

1. Resolve a checkpoint path from a checkpoint file or pretrained keyword.
2. Build the model, optionally selecting one tensor-product accelerator backend.
3. Load targets as raw structures, `structure_list`, or existing graph datasets.
4. If raw structures are used, build graphs for the current snapshot set.
5. Run batched forward evaluation.
6. Convert batched outputs to per-structure and per-atom records and write result files.

Standard output directory contents:

| File | Content |
| --- | --- |
| `errors.txt` | Aggregate error metrics over labeled targets |
| `info.csv` | Per-structure metadata such as source file paths |
| `per_graph.csv` | Per-structure references and predictions |
| `per_atom.csv` | Per-atom positions, references, predictions, and atomic energies |
| `sevenn_data/saved_graph.pt` and `.yaml` | Optional saved preprocessed graphs when `--save_graph` is used |

Stable inference behavior:

- `device=auto` selects CUDA when available, otherwise CPU.
- `--modal` is required for multi-modal checkpoints.
- Stress is written in kbar in the CSV/error outputs.
- Existing `.pt` graph datasets skip graph construction and are the stable way to avoid repeated preprocessing.

Performance considerations:

- Use prebuilt `.pt` datasets when the same fixed snapshot set is evaluated repeatedly.
- Use batching on GPU inference to improve device utilization.
- Accelerator backends only affect the equivariant convolution path after geometry is available.
- Experimental pairgeom only reuses pair-invariant work within one snapshot and does not make topology sufficient to replace geometry.

(accelerators)=
# 6. Accelerators
All three stable accelerators in SevenNet target tensor-product-heavy parts of the equivariant convolution path. None of them changes the mathematical model architecture, neighbor topology definition, or the requirement to form geometry from the current snapshot.

Backend scope by concept:

| Concept | Stable meaning | Affected by FlashTP / cuEquivariance / OpenEquivariance |
| --- | --- | --- |
| Topology | `edge_index`, `pbc_shift`, graph connectivity | No |
| Geometry | `edge_vec`, distances, radial basis, cutoff, spherical harmonics | No |
| Convolution compute | Learned filter application, tensor products, and some scatter/gather work depending on backend | Yes |

Backend-specific behavior:

| Backend | What it accelerates | Stable user surfaces | Important limits |
| --- | --- | --- | --- |
| FlashTP | Fused tensor-product convolution kernels, with fused scatter/gather in the patched path | Training, checkpoint inference, ASE calculator, TorchScript LAMMPS export, ML-IAP export | GPU-only; does not build geometry; not a model-architecture change |
| cuEquivariance | Convolution/tensor-product path for checkpoint models | Training, checkpoint inference, ASE calculator, ML-IAP export | GPU-only; current SevenNet integration does not support TorchScript LAMMPS export with cuEquivariance |
| OpenEquivariance | Compiled tensor-product convolution path | Training, checkpoint inference, ASE calculator, TorchScript LAMMPS export, ML-IAP export | GPU-only; still a convolution backend, not a geometry backend |

Current implementation notes that matter for correctness:

- FlashTP patches convolution modules only.
- cuEquivariance patches the convolution path in current SevenNet integration; it is not the model definition itself.
- OpenEquivariance patches the convolution path through a compiled tensor-product-convolution wrapper.
- In stable usage, enable at most one accelerator backend at a time.

(pairgeom-experimental)=
# 7. Experimental Features
## 7.1 `pairgeom` (EXPERIMENTAL)

`pairgeom` is experimental, development-stage functionality. It is not part of the stable deployment surface.

Supported scope:

- Inference only
- Checkpoint-backed runtime only
- Available through checkpoint inference and checkpoint-based ASE calculator use

Not supported in the stable surface:

- Training CLI
- `sevenn get_model`
- TorchScript deployment artifacts
- LAMMPS deployment

What `pairgeom` reuses within one forward pass of one snapshot:

- Radial basis values
- Cutoff-function values
- Spherical harmonics for a canonical pair direction, with reverse-direction parity handling derived from that pair value
- `weight_nn` outputs that depend only on pair-invariant radial input

What `pairgeom` does not reuse:

- Tensor-product evaluation
- Directional message passing
- Message aggregation

Operational constraints:

- The graph must contain reverse directed edges.
- Periodic-image pairing requires consistent `pbc_shift` bookkeeping.
- The implementation assumes exactly two directed edges per undirected pair.
- Zero-shift self edges are not supported by the pair-mapping routine.

Correct scope of reuse:

- Reuse is within the same forward pass of the same snapshot.
- Reuse across different timesteps or different snapshots is generally invalid because geometry is position-dependent.
- If positions change, distances, spherical harmonics, and all other geometry-derived quantities must be recomputed from the new snapshot.
- Preserving force/stress correctness requires keeping the current-geometry autograd path intact.

Current execution paths:

- Serial e3nn inference can use a reference pair-fused convolution path.
- FlashTP, cuEquivariance, and OpenEquivariance backends still execute their directed-edge tensor-product path even when pairgeom is enabled.

Performance implication:

- The benefit depends on how much runtime is spent on pair-invariant geometry and pair-invariant filter generation relative to tensor-product work.
- If tensor products dominate, expected gains are limited.
- If geometry/filter preparation is a significant fraction of runtime, gains can be larger.

Additional note for ASE:

- The ASE calculator may cache pair metadata such as pair ownership when `edge_index` and `pbc_shift` are unchanged.
- That cache is topology metadata reuse only. It is not timestep-to-timestep geometry reuse.
