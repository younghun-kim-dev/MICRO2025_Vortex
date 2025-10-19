#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// 출력 C 행렬의 타입을 명시적으로 int32로 고정
#ifndef TYPE
#define TYPE int32_t
#endif

typedef struct {
  uint32_t grid_dim[2];  // grid_dim[0]=cols(N), grid_dim[1]=rows(N)
  uint32_t size;         // N
  uint64_t A_addr;       // device VA for A (int8_t[N*N])
  uint64_t B_addr;       // device VA for B (int8_t[N*N])
  uint64_t C_addr;       // device VA for C (int32_t[N*N])
} kernel_arg_t;

#endif

