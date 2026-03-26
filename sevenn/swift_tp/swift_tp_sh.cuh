#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ void compute_real_sh_l3_cartesian(
    float dx,
    float dy,
    float dz,
    float r,
    float sh[16]
) {
  constexpr float kEps = 1.0e-12f;
  constexpr float kSqrt3 = 1.7320508075688772f;
  constexpr float kSqrt5 = 2.2360679774997898f;
  constexpr float kSqrt7 = 2.6457513110645907f;
  constexpr float kSqrt15 = 3.8729833462074170f;
  constexpr float kSqrt42 = 6.4807406984078602f;
  constexpr float kSqrt168 = 12.961481396815721f;

  const float rr = (r > 0.0f) ? r : sqrtf(dx * dx + dy * dy + dz * dz);
  const float inv_r = rsqrtf(fmaxf(rr * rr, kEps));
  const float x = dx * inv_r;
  const float y = dy * inv_r;
  const float z = dz * inv_r;

  const float y2 = y * y;
  const float x2z2 = x * x + z * z;

  sh[0] = 1.0f;
  sh[1] = kSqrt3 * x;
  sh[2] = kSqrt3 * y;
  sh[3] = kSqrt3 * z;

  sh[4] = kSqrt15 * x * z;
  sh[5] = kSqrt15 * x * y;
  sh[6] = kSqrt5 * (y2 - 0.5f * x2z2);
  sh[7] = kSqrt15 * y * z;
  sh[8] = 0.5f * kSqrt15 * (z * z - x * x);

  sh[9] = (kSqrt42 / 6.0f) * (sh[4] * z + sh[8] * x);
  sh[10] = kSqrt7 * sh[4] * y;
  sh[11] = (kSqrt168 / 8.0f) * ((4.0f * y2 - x2z2) * x);
  sh[12] = 0.5f * kSqrt7 * y * (2.0f * y2 - 3.0f * x2z2);
  sh[13] = (kSqrt168 / 8.0f) * (z * (4.0f * y2 - x2z2));
  sh[14] = kSqrt7 * sh[8] * y;
  sh[15] = (kSqrt42 / 6.0f) * (sh[8] * z - sh[4] * x);
}
