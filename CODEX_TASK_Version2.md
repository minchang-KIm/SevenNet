# Codex Implementation Task: Pair-Aware Geometry Reuse + Option Plumbing (SevenNet)

**Target repo:** `MDIL-SNU/SevenNet`  
**Goal:** Implement *pair-aware geometry reuse* for equivariant MLIP inference in SevenNet, add a runtime option to enable it, make it compatible with existing accelerator options (FlashTP / cuEquivariance / OpenEquivariance), and provide validation + benchmarking automation.

This task is **performance-critical** and must preserve **numerical semantics** (energy/force/stress within tolerance).  
This task must be implemented in a way that is **reversible** (baseline path remains available).

---

## 0. Non-Negotiable Constraints

### MUST NOT change
- model definition / tensor product math / irreps definitions
- training semantics (loss, data, correctness)
- node-centric aggregation semantics (messages still aggregated to destination nodes)

### MUST change ONLY
- geometry/filter preparation path (distance / radial basis / radial embedding / spherical harmonics)
- internal scheduling of geometry computations
- optional reduction scheduling **only if** it is provably correct and stable

### Compatibility requirements
- Must work with baseline e3nn path
- Must work when `FlashTP` is enabled (`use_flash_tp=True`)
- Must work when `cuEquivariance` is enabled (`cuequivariance_config.use=True`)
- Must work when `OpenEquivariance` is enabled (`use_oeq=True`)
- Must not break TorchScript export if the project supports it (if TorchScript exists, add a test)

---

## 1. What “Pair-Aware Geometry Reuse” Means (Implementation Definition)

SevenNet currently uses a **directed edge list**; each undirected pair (i,j) appears twice:
- edge e1: i -> j
- edge e2: j -> i

In baseline, geometry-dependent tensors are computed per directed edge:
- displacement vectors / distances
- radial basis / radial embedding (radial MLP outputs)
- spherical harmonics (SH)

### Pair-aware optimization
Compute geometry once for **canonical undirected pairs**:
- canonical pair: `(min(i,j), max(i,j))`

Create mapping:
- `edge_to_pair[e]` = index of undirected pair for directed edge e
- `edge_is_reversed[e]` = whether directed edge is opposite to canonical orientation

Reuse:
- distance / radial basis / radial embedding can be reused directly
- spherical harmonics can be reused with **parity transform** for reversed edges:

For unit direction `r_hat`, reversed direction is `-r_hat`:
- `Y_l(-r_hat) = (-1)^l * Y_l(r_hat)`

Therefore apply a **degree-parity sign** to SH components for reversed edges.

**Important:** Do not reuse full messages or tensor products. Only geometry/filter preparation.

---

## 2. Repo Recon: What to Locate (Codex must discover and cite file/line locations)

Codex MUST first find the exact locations where SevenNet computes:
1. `edge_index` and `edge_vectors` / `edge_vec` (neighbor list -> directed edges)
2. Geometry preparation:
   - distances
   - radial basis / radial embedding
   - spherical harmonics
3. Where these tensors are fed into convolution / tensor product
4. Accelerator patching:
   - FlashTP patch (`sevenn/nn/flash_helper.py`, `sevenn/model_build.py::patch_flash_tp`)
   - cuEq patch (`sevenn/model_build.py::patch_cue`)
   - OEQ patch (helper + patch function)

Codex MUST create a short **“system map”** section in the PR description or notes:
- entrypoint(s) for inference
- entrypoint(s) for training
- the exact call chain from `data[EDGE_VEC]` to SH/radial to convolution

---

## 3. New User-Facing Option: `use_pairaware`

### 3.1 Add config key
Add to `sevenn/_keys.py`:
- `USE_PAIRAWARE = "use_pairaware"`

### 3.2 Default behavior
- Default: `False` (baseline path)
- When enabled: pair-aware geometry reuse path

### 3.3 CLI plumbing (must be consistent with existing accelerator options)
Add CLI flag:
- Preferred: `--enable_pairaware` (mirrors `--enable_flash`, `--enable_oeq`, etc.)
- Or: `--use_pairaware` (but keep consistent naming across commands)

Where to wire:
- `sevenn/main/sevenn.py` (it already maps `--enable_flash` into `model_config[USE_FLASH_TP]=True`)
- Any subcommand that builds models or exports deployed models (e.g. `sevenn get_model`, `sevenn inference`, etc.)

**Requirement:** the option must work for:
- pure python inference
- training
- model export (if exists)

### 3.4 Logging / Reporting
Whenever a model is built or inference is run, print to screen/log:
- `use_pairaware=<bool>`
- `use_flash_tp=<bool>`
- `cuequivariance=<bool>` + relevant subconfig
- `use_oeq=<bool>`
- plus the resolved “effective mode” (baseline / pairaware / flashtp / combined)
- if pair-aware is enabled, also print:
  - `num_edges_directed`
  - `num_pairs_undirected`
  - estimated geometry reuse factor: `num_edges_directed / num_pairs_undirected`

This print must happen *every inference run* (and training start), so users can confirm the mode.

---

## 4. Core Implementation Plan (Step-by-Step)

### Phase A — Build undirected pairs + mapping
Given directed edges `edge_index` shape [2, E] (src, dst):
1. compute `pair_src = min(src, dst)`
2. compute `pair_dst = max(src, dst)`
3. create a unique key for each pair:
   - safest: `key = pair_src * num_nodes + pair_dst` (requires num_nodes known)
   - alternative: `torch.stack([pair_src, pair_dst], dim=0)` + `unique(dim=1, return_inverse=True)` if stable

Output:
- `pair_index` shape [2, P] canonical pairs
- `edge_to_pair` shape [E] (int64)
- `edge_is_reversed` shape [E] (bool): `src > dst` (or `src != pair_src`)

**Performance requirements**
- Must be GPU-friendly (avoid Python loops)
- Must use stable torch ops (avoid extremely slow unique for giant E if possible)
- If `torch.unique` is used, add a note and benchmark its overhead

### Phase B — Pair-level geometry compute
Compute once per pair:
- displacement vector `r_ij = pos[pair_dst] - pos[pair_src]` (+ PBC shift if applicable)
- distance `d = ||r||`
- unit direction `r_hat`
- spherical harmonics `Y(pair)` once for canonical direction
- radial basis / radial embedding once

Store in buffers indexed by pair id:
- `pair_dist: [P]`
- `pair_radial: [...]`
- `pair_sh: [...]`

### Phase C — Directed edge view via mapping
To feed existing directed-edge code without rewriting:
- `edge_dist = pair_dist[edge_to_pair]`
- `edge_radial = pair_radial[edge_to_pair]`
- `edge_sh = pair_sh[edge_to_pair]` then apply parity for reversed edges

#### Parity transform implementation detail (critical)
You must determine how SevenNet represents SH:
- it is usually concatenated by irreps degrees l; signs differ per l
- Implement a function that multiplies the SH channels corresponding to odd l by -1 **only when edge_is_reversed is true**.

**How to do it correctly**
- Build a precomputed sign vector `sh_parity_sign` of shape `[sh_dim]`:
  - entries are +1 for even-l components, -1 for odd-l components
- Then:
  - `edge_sh = edge_sh * where(edge_is_reversed[:,None], sh_parity_sign[None,:], 1)`

The hard part: mapping “channels -> l parity”.
Codex MUST:
- locate where SH is computed (it will know lmax / irreps)
- construct the sign vector using the same ordering that SH uses
- add unit tests verifying parity behavior using a simple synthetic direction vector

### Phase D — Integration point: where to plug this
Codex must find the function/module that creates:
- `EDGE_VEC` and derived `EDGE_LENGTH`, radial basis, SH tensors

Then add:
- `if config[USE_PAIRAWARE]:` call pair-aware geometry builder
- else baseline builder

**Keep output tensor shapes identical** to baseline so downstream convolution works.

### Phase E — FlashTP/cuEq/OEQ compatibility
Because these accelerators patch the convolution kernel, not the geometry tensors, compatibility should hold if:
- geometry tensors remain identical shape/type
- optional cached buffers do not break autograd

Codex MUST verify:
- baseline + FlashTP
- pairaware + FlashTP
- baseline + cuEq
- pairaware + cuEq (if cuEq requires certain layout, confirm)
- baseline + OEQ
- pairaware + OEQ

If any accelerator bypasses the baseline geometry builder, add hooks there too.

---

## 5. Memory & Performance Design (Think deeply; implement consciously)

### 5.1 What to cache and where
- Pair-level buffers `pair_*` live on the same device as model inference (CUDA).
- They should be **contiguous** and prefer FP32 for geometry (match existing convention; do not change precision).
- Avoid storing both edge-level and pair-level SH simultaneously if memory is tight:
  - compute `edge_sh` as a view/indexed tensor + sign multiply; do not keep an extra copy if not needed

### 5.2 Avoiding costly ops
- Pair construction with `torch.unique` can be expensive.
  - If E is huge, consider an O(E log E) sort-based canonicalization:
    - sort pairs by key, compute segment ids, map edges -> pair via inverse indices
- DO NOT introduce Python loops over edges/pairs.

### 5.3 Autograd correctness
- Geometry buffers may require gradients if forces are computed via `autograd` on positions/edge_vec.
- Ensure that pair-aware path still allows gradients to flow correctly:
  - indexing operations are differentiable
  - parity multiplication is differentiable
  - no accidental `.detach()` or `.no_grad()`

### 5.4 Threading / stream safety
- Keep torch ops on current stream.
- Avoid host-device sync (no `.cpu()` during inference).

---

## 6. Validation Plan (Codex MUST implement these checks)

### 6.1 Unit tests (fast)
Add tests that run in CI CPU-only if possible; GPU optional.

1) **Parity correctness test**
- Build a small synthetic graph with a pair direction r and reversed -r.
- Compute SH baseline for both directions.
- Compute pair-aware SH using parity transform.
- Assert they match within tolerance.

2) **Mapping correctness test**
- Create directed edges for a simple triangle or square; verify:
  - undirected pair count is correct
  - each pair maps to exactly 2 directed edges (unless self-edge or boundary cases)
  - `edge_is_reversed` is correct

### 6.2 Golden numerical tests (model-level)
Pick a tiny pretrained model or a tiny random model config (if repo has a toy model).
For a fixed seed and structure:
- run baseline inference: energy/forces/stress
- run pair-aware inference: compare
- acceptance thresholds:
  - energies: absolute diff < 1e-6 (or close to baseline’s numeric noise)
  - forces: max abs diff < 1e-6 to 1e-5 (depending on backend)
If accelerators produce slightly different floating behavior, widen tolerance but document it.

### 6.3 Integration tests for accelerators
If FlashTP is not installed in CI, still test “fallback logic does not break”:
- enabling flash without package should raise the expected error or warning (consistent with existing behavior)

If GPU CI is available:
- run the 4-mode matrix:
  - baseline
  - pairaware
  - flashtp
  - combined

### 6.4 Runtime self-reporting
Every inference execution should print:
- effective mode
- directed edges E
- undirected pairs P
- geometry time and total time (if profiling enabled)

---

## 7. Benchmarking (Deliverable: runnable shell + python harness)

### 7.1 Add CLI for benchmarking
Add a script (examples/bench or tools/) that:
- loads a structure or generates a random dense structure
- runs N warmup steps + M timed steps
- prints:
  - avg step time
  - geometry time
  - tensor product time (if possible)
  - total time
  - throughput

### 7.2 Provide a shell runner
Create `bench/run_bench.sh` that runs:
1) baseline
2) pairaware
3) flashtp
4) combined

Also run with multiple sizes:
- small (~256 atoms)
- medium (~2k atoms)
- large (~20k atoms) if feasible

### 7.3 Profiling hooks (optional)
- If `torch.profiler` is available, add an option `--profile` that writes chrome traces.

---

## 8. “Codex Self-Verification Loop” Requirements

Codex MUST follow this loop until all checks pass:

1. Implement changes.
2. Run unit tests.
3. Run golden inference compare baseline vs pair-aware.
4. If mismatch:
   - print a diagnosis summary
   - identify which tensor diverged first (distance/radial/SH/tp output)
   - fix and repeat
5. Run benchmark and confirm speedup in geometry phase.
6. Confirm FlashTP compatibility by running the combined mode when available.

**Codex MUST report to the console** at each inference run:
- mode flags (pairaware/flash/cueq/oeq)
- E and P counts
- timing summary

If any step fails, Codex MUST:
- update this document’s “Known Issues” section (append)
- implement the fix
- rerun the failing step

---

## 9. Deliverables Checklist (What must be in the final PR)

### Code
- [ ] `use_pairaware` config key added and used
- [ ] pair-aware geometry implementation + baseline fallback
- [ ] parity transform implemented correctly and tested
- [ ] CLI flag wired in all relevant entrypoints
- [ ] logging/reporting shows effective mode

### Tests
- [ ] mapping test
- [ ] parity SH test
- [ ] model-level numerical equivalence test (baseline vs pairaware)

### Bench
- [ ] benchmark python script
- [ ] benchmark shell runner `run_bench.sh`
- [ ] output includes timings for geometry/total and E/P stats

### Docs
- [ ] short doc update: how to enable `--enable_pairaware`
- [ ] note that it composes with FlashTP/cuEq/OEQ

---

## 10. Notes / Known Issues (append during implementation)
- 2026-03-19: Validation on this workspace confirmed CPU correctness but not the CPU performance target. Pair-aware numerical tests, CLI/export checks, and profile output checks passed after installing the missing Python deps into `/tmp/sevenn-pairaware-venv`.
- 2026-03-19: `bench/pairaware_bench.py` had a profile-path bug (`runtime_config` was undefined). It was fixed during validation, and the harness now also checks requested `pairaware` activation explicitly.
- 2026-03-19: CPU benchmark results show the geometry stage regressing instead of improving with pair-aware enabled:
  - 256 atoms: `67.588 ms -> 78.585 ms`
  - 2000 atoms: `65.571 ms -> 123.951 ms`
  - 21296 atoms: `53.144 ms -> 773.454 ms`
- 2026-03-19: The current likely bottleneck is pair construction via `torch.unique(..., dim=0, return_inverse=True)` in `sevenn/nn/edge_embedding.py`. Correctness is strong, but the current CPU implementation is not performance-positive.
- 2026-03-19: FlashTP / cuEq / OpenEquivariance remain unverified on this machine because `torch.cuda.is_available()` is `False`, and all three availability probes returned `False`; the corresponding test suites skipped cleanly.
