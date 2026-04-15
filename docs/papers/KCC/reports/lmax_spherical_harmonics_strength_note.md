# lmax와 Spherical Harmonics 재사용 메모

이 문서는 논문 본문을 다시 정리하지 않고, `lmax`가 왜 중요한지와 현재 제안기법이 이 축에서 어디에 강점을 가질 수 있는지를 정리해 둔 작업 메모다.

## 1. 공식 문서 기준으로 보통 어떤 `lmax`를 쓰는가

- NequIP 공식 문서는 `l_max=1`을 좋은 기본값으로 소개하고, `l_max=2`는 더 정확하지만 더 느리다고 설명한다. 또한 foundation preset은 `S/M/L/XL = l_max 1/2/3/4`로 제공한다.  
  출처: https://nequip.readthedocs.io/en/latest/_modules/nequip/model/nequip_models.html
- SevenNet 공식 문서는 `SevenNet-0`이 `lmax=2`, `SevenNet-l3i5`가 `lmax=3`이라고 밝히며, `l3i5`는 정확도는 좋아지지만 `SevenNet-0`보다 약 4배 느리다고 설명한다. 최신 Omni/MF/OMAT 계열도 모두 `lmax=3`을 사용한다.  
  출처: https://github.com/MDIL-SNU/SevenNet/blob/main/docs/source/user_guide/pretrained.md

정리하면, 실무적으로는 `1`이 아주 가벼운 기본값이고, `2`와 `3`이 더 흔한 정확도 지향 설정이며, `4`는 큰 모델에서만 쓰는 편이라고 보는 것이 맞다.

## 2. 이론적으로 `l`이 커지면 무엇이 좋아지고 무엇이 비싸지는가

- `lmax`는 각 edge에서 표현할 수 있는 각도 정보의 해상도를 정한다.
- `l=0`은 방향성이 없는 스칼라 성분이다.
- `l=1`은 1차 방향 정보를 담는다.
- `l=2`, `l=3`로 갈수록 더 복잡한 각도 변화를 표현할 수 있다.
- 따라서 방향성 결합이 강한 공유결합, 분자, 표면 흡착, 저대칭 결정 구조에서는 더 높은 `l`이 도움이 될 수 있다.

반대로 비용도 커진다.

- spherical harmonics 출력 차원은 `sum_(l=0)^L (2l+1) = (L+1)^2`로 증가한다.
- 즉 `lmax=1,2,3,4`에서 SH 차원은 `4, 9, 16, 25`로 커진다.
- 이 증가는 SH 단계뿐 아니라, 이후 irreps와 tensor product 경로 수, 파라미터 수 증가로 이어진다.
- 따라서 이론적으로도 `최적의 lmax`는 하나로 고정되지 않는다. 각도 표현력이 필요한 문제에서는 올리는 것이 이득이고, 그렇지 않으면 비용만 커진다.

## 3. 로컬 구조 확인

| lmax | SH dim | TP instructions (1st conv) | TP weight numel | Trainable params |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4 | 2 | 8 | 16698 |
| 2 | 9 | 3 | 12 | 20282 |
| 3 | 16 | 4 | 16 | 26378 |
| 4 | 25 | 5 | 20 | 35802 |

이 표는 같은 SevenNet 구조에서 `lmax`만 바꿨을 때의 변화다. 적어도 현재 코드 기준으로는 `lmax`가 올라갈수록 SH 차원은 정확히 제곱 꼴로 커지고, 파라미터 수도 함께 증가한다.

## 4. 로컬 timing 검증

측정 환경:

- device: `cuda`
- warmup: `20`
- repeat: `100`
- small graph: benzene (`114` directed edges)
- large graph: NaCl `10x10x10` supercell (`36000` directed edges)

측정한 값:

1. `baseline_sh`: 모든 directed edge에 대해 SH 직접 계산
2. `pair_sh_kernel_only`: pair 기준 SH 한 번만 계산
3. `pair_sh`: pair 기준 SH 한 번 계산 후 reverse는 sign flip으로 복원
4. `baseline_edge_embedding`: 기존 edge embedding
5. `pair_edge_embedding`: pair-aware geometry-only edge embedding

pair 적용 속도비:

| graph | lmax | SH kernel-only speedup | SH reconstructed speedup | Edge embedding speedup |
| --- | ---: | ---: | ---: | ---: |
| large_nacl_10x10x10 | 1 | 1.012 | 0.701 | 0.620 |
| large_nacl_10x10x10 | 2 | 1.022 | 0.835 | 0.698 |
| large_nacl_10x10x10 | 3 | 1.041 | 0.946 | 0.781 |
| large_nacl_10x10x10 | 4 | 1.042 | 0.975 | 0.866 |
| small_benzene | 1 | 1.014 | 0.700 | 0.645 |
| small_benzene | 2 | 1.013 | 0.817 | 0.697 |
| small_benzene | 3 | 0.989 | 0.883 | 0.768 |
| small_benzene | 4 | 0.991 | 0.949 | 0.846 |

## 5. 이 메모에서 바로 가져갈 수 있는 메시지

1. `lmax`가 커질수록 SH가 다루는 성분 수는 빠르게 증가한다.
2. 따라서 SH를 정확히 반으로 줄이는 pair reuse의 절대 절감량도 커질 가능성이 높다.
3. 최신 SevenNet 계열이 `lmax=3`을 쓰고 있다는 점은, 이 방향이 실제로 중요한 영역에 이미 들어와 있다는 뜻이다.
4. 즉 현재 제안기법의 강점은 단순히 "SH를 한 번 덜 계산한다"가 아니라, `lmax`가 커질수록 더 비싸지는 각도 표현 비용을 정확도 변화 없이 줄인다는 점이다.

## 6. 주의할 점

- 이 메모는 아직 논문 본문용으로 다듬지 않았다.
- 현재 timing은 SH/edge-embedding 단계의 국소 실험이다.
- 전체 force-including MD step 성능은 backward, pair metadata, full path 구조에 더 크게 좌우된다.
- 따라서 여기서 바로 "전체 모델이 lmax가 높을수록 pair가 더 빠르다"라고 주장하면 안 된다.
- 대신 "SH 중심의 geometry-side 절감 강도는 lmax가 커질수록 더 중요해진다"까지는 방어 가능하다.
