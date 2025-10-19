#include <stdint.h>
#include <vx_spawn.h>
#include "vx_intrinsics.h"
#include "common.h"

// 안전한 패킹(언얼라인드 포인터 금지): 4×int8 -> u32 (LSB=x0, MSB=x3)
static inline uint32_t pack_i8x4(int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
  return  (uint32_t)(uint8_t)x0
        | ((uint32_t)(uint8_t)x1 << 8)
        | ((uint32_t)(uint8_t)x2 << 16)
        | ((uint32_t)(uint8_t)x3 << 24);
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  const auto* A = reinterpret_cast<const int8_t*>(arg->A_addr);
  const auto* B = reinterpret_cast<const int8_t*>(arg->B_addr);
        auto* C = reinterpret_cast<int32_t*>(arg->C_addr);
  const int N  = (int)arg->size;

  // sgemm 템플릿과 동일한 블록 인덱스 사용
  const int col = blockIdx.x;
  const int row = blockIdx.y;

  int32_t acc = 0;

  // 4원씩 VX_DOT8 누산
  int k = 0;
  for (; k + 3 < N; k += 4) {
    const uint32_t packedA = pack_i8x4(
      A[row * N + (k + 0)],
      A[row * N + (k + 1)],
      A[row * N + (k + 2)],
      A[row * N + (k + 3)]
    );
    const uint32_t packedB = pack_i8x4(
      B[(k + 0) * N + col],
      B[(k + 1) * N + col],
      B[(k + 2) * N + col],
      B[(k + 3) * N + col]
    );
    acc += vx_dot8((int)packedA, (int)packedB);
  }

  // N%4 꼬리 처리(스칼라)
  for (; k < N; ++k) {
    acc += (int)A[row * N + k] * (int)B[k * N + col];
  }

  C[row * N + col] = acc;
}

int main() {
  // mscratch로 전달된 커널 인자 로드
  auto* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

  // sgemm 테스트와 동일한 런타임 스폰 방식 (2D grid)
  return vx_spawn_threads(
    /*num_cores=*/2,            // 환경에 맞게 유지
    arg->grid_dim,              // grid_dim[0]=N, grid_dim[1]=N
    nullptr,
    (vx_kernel_func_cb)kernel_body,
    arg
  );
}

