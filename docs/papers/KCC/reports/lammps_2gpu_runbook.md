# SevenNet LAMMPS 2-GPU 실행 절차 메모

이 문서는 현재 코드 기준으로 `LAMMPS + e3gnn/parallel` 경로에서 SevenNet baseline과 제안기법을 비교하기 위한 실행 절차를 정리한 메모다.

## 1. 전제

- 공식 사용 문서: [lammps_torch.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/source/user_guide/lammps_torch.md)
- 병렬 pair style: `e3gnn/parallel`
- 기본 가정: **MPI rank당 GPU 1개**
- 현재 이 세션에서 보이는 GPU는 1개뿐이다.
  - `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader`
  - 결과: `0, NVIDIA GeForce RTX 4090, 24564 MiB`
- 따라서 여기서는 2-GPU 실행을 직접 검증하지 못했고, 아래 절차는 네 클러스터에서 실행해야 한다.

## 2. LAMMPS 빌드

문서 요구 버전은 `stable_2Aug2023_update3`다.

```bash
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_2Aug2023_update3 --depth=1
sevenn patch_lammps ./lammps_sevenn
```

FlashTP를 쓰지 않는 baseline/proposal 비교라면 `--enable_flash`는 넣지 않는다.

그다음 빌드:

```bash
cd lammps_sevenn
mkdir -p build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
make -j4
```

실행 파일 확인:

```bash
./lmp -help | head
```

## 3. 병렬 deploy 모델 생성

### 3-1. baseline

예시는 `7net-omni`, `modal=mpa` 기준이다.

```bash
sevenn get_model 7net-omni \
  --get_parallel \
  --modal mpa \
  -o deployed_parallel_baseline
```

### 3-2. proposal: geometry_only

현재 코드/실험 기준으로 메인 제안 비교는 `geometry_only`가 더 타당하다.  
`full`은 아직 pair-major가 아니고, 구조적으로 느린 상태다.

```bash
sevenn get_model 7net-omni \
  --get_parallel \
  --modal mpa \
  --enable_pair_execution \
  --pair_execution_policy geometry_only \
  -o deployed_parallel_pair_geo
```

### 3-3. ablation: full

```bash
sevenn get_model 7net-omni \
  --get_parallel \
  --modal mpa \
  --enable_pair_execution \
  --pair_execution_policy full \
  -o deployed_parallel_pair_full
```

### 3-4. 병렬 모델 sanity check

각 디렉터리에 `deployed_parallel_0.pt`, `deployed_parallel_1.pt`, ... 가 있어야 한다.

```bash
ls deployed_parallel_baseline
ls deployed_parallel_pair_geo
ls deployed_parallel_pair_full
```

## 4. LAMMPS 입력 파일

최소 입력 예시는 아래와 같다.

```text
units       metal
atom_style  atomic
boundary    p p p
read_data   system.data

pair_style  e3gnn/parallel
pair_coeff  * * 4 ./deployed_parallel_baseline Hf O

timestep    0.001
thermo      10
run         100
```

여기서 `4`는 message-passing layer 수다.  
실제 layer 수는 deploy된 `*.pt` 파일 개수와 일치해야 한다.

제안기법 실행은 디렉터리만 바꾸면 된다.

```text
pair_coeff  * * 4 ./deployed_parallel_pair_geo Hf O
```

## 5. 2-GPU 실행

기본 형태:

```bash
mpirun -np 2 ./lmp -in in.parallel_baseline.lmp
mpirun -np 2 ./lmp -in in.parallel_pair_geo.lmp
```

중요한 점:

- rank 2개면 GPU도 2개가 보이도록 잡아야 한다.
- 일반적으로는:

```bash
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ./lmp -in in.parallel_baseline.lmp
```

- `one GPU per MPI rank`를 가정한다.

## 6. 권장 비교 순서

논문용으로는 아래 순서를 권장한다.

1. `baseline`
2. `geometry_only`
3. `full`은 보조 ablation

이유:

- 현재 실험상 `geometry_only`는 제안기법의 순수 geometry-side 효과를 더 잘 보여준다.
- `full`은 아직 pair-major가 아니므로, 메인 주장을 흐릴 가능성이 크다.

## 7. 측정 권장 항목

최소한 아래는 저장해야 한다.

- wall-clock step time
- atoms 수
- edge 수
- 평균 neighbor 수
- energy 차이
- force 차이
- GPU utilization

가능하면 `fix balance`를 켜고 안 켜고도 같이 본다.

## 8. 주의점

- 문서에도 적혀 있듯이, parallel path는 빈 subdomain이 생기면 에러가 날 수 있다.
- 따라서 `processors` 또는 `fix balance`를 쓰는 편이 안전하다.
- 현재 main 세션에서는 GPU가 1개만 보여서 이 절차를 여기서 직접 검증하지 못했다.

## 9. 현재 코드 기준 추천

- 메인 비교: `baseline vs geometry_only`
- `full`은 보조 실험
- FlashTP는 후속 실험으로 분리

지금 단계에서 논문 메시지를 가장 깔끔하게 유지하려면 이 구성이 맞다.
