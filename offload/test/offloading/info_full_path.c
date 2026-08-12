// RUN: %libomptarget-compile-generic \
// RUN:     -gline-tables-only -fopenmp-extensions
// RUN: env LIBOMPTARGET_INFO=63 LIBOMPTARGET_INFO_FULL_PATH=1 \
// RUN:   %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -allow-empty -check-prefixes=INFO

// FIXME: Fails due to optimized debugging in 'ptxas'.
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO

#include <stdio.h>

#define N 64

int main() {
  int A[N];

// With LIBOMPTARGET_INFO_FULL_PATH=1 the runtime must keep the full source
// path (the leading directory is present), instead of reducing it to the
// basename "info_full_path.c".
// clang-format off
// INFO: info: Entering OpenMP kernel at {{.*}}/info_full_path.c:{{[0-9]+}}:{{[0-9]+}} with 1 arguments:
// clang-format on
#pragma omp target map(tofrom : A[0 : N])
  {
    for (int i = 0; i < N; ++i)
      A[i] = i;
  }

  return 0;
}
