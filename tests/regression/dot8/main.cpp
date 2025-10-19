#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cstdlib>
#include "common.h"

#define RT_CHECK(_expr) do {                               \
  int _ret = (_expr);                                      \
  if (0 == _ret) break;                                    \
  std::printf("Error: '%s' returned %d!\n", #_expr, _ret); \
  cleanup();                                               \
  std::exit(-1);                                           \
} while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t size = 32;  // N

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex dot8 GEMM test\n"
            << "Usage: [-k kernel] [-n size] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
      case 'n': size = std::atoi(optarg); break;
      case 'k': kernel_file = optarg;     break;
      case 'h': show_usage(); std::exit(0);
      default:  show_usage(); std::exit(-1);
    }
  }
}

static void matmul_cpu_int8_int32(int32_t* C, const int8_t* A, const int8_t* B, uint32_t N) {
  for (uint32_t i = 0; i < N; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (uint32_t k = 0; k < N; ++k) {
        acc += (int)A[i*N + k] * (int)B[k*N + j];
      }
      C[i*N + j] = acc;
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
    device = nullptr;
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  std::cout << "open device connection\n";
  RT_CHECK(vx_dev_open(&device));

  const uint32_t N = size;
  const uint32_t NN = N * N;

  // 버퍼 바이트 크기: A,B=int8, C=int32
  const uint32_t bytes_A = NN * sizeof(int8_t);
  const uint32_t bytes_B = NN * sizeof(int8_t);
  const uint32_t bytes_C = NN * sizeof(int32_t);

  std::cout << "matrix size: " << N << "x" << N << "\n";

  kernel_arg.grid_dim[0] = N;  // col
  kernel_arg.grid_dim[1] = N;  // row
  kernel_arg.size        = N;

  // 디바이스 메모리 할당
  std::cout << "allocate device memory\n";
  RT_CHECK(vx_mem_alloc(device, bytes_A, VX_MEM_READ,  &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, bytes_B, VX_MEM_READ,  &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, bytes_C, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << "\n";
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << "\n";
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::dec << "\n";

  // 호스트 데이터 준비 (A,B는 int8, C는 int32)
  std::vector<int8_t>  h_A(NN);
  std::vector<int8_t>  h_B(NN);
  std::vector<int32_t> h_C(NN, 0);

  // 값 범위를 작게(오버플로우 방지)
  for (uint32_t i = 0; i < NN; ++i) h_A[i] = (int8_t)((std::rand() % 7) - 3); // [-3..3]
  for (uint32_t i = 0; i < NN; ++i) h_B[i] = (int8_t)((std::rand() % 5) - 2); // [-2..2]

  // 업로드
  std::cout << "upload matrix A/B buffers\n";
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, bytes_A));
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, bytes_B));

  // 커널 바이너리 업로드
  std::cout << "upload kernel binary\n";
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // 커널 인자 업로드
  std::cout << "upload kernel argument\n";
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // 실행
  auto t0 = std::chrono::high_resolution_clock::now();
  std::cout << "start device\n";
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  std::cout << "wait for completion\n";
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::printf("Elapsed time: %g ms\n", ms);

  // 결과 다운로드
  std::cout << "download destination buffer\n";
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, bytes_C));

  // 검증
  std::cout << "verify result\n";
  std::vector<int32_t> h_ref(NN);
  matmul_cpu_int8_int32(h_ref.data(), h_A.data(), h_B.data(), N);

  int errors = 0;
  for (uint32_t i = 0; i < NN; ++i) {
    if (h_C[i] != h_ref[i]) {
      if (errors < 100) {
        uint32_t r = i / N, c = i % N;
        std::printf("*** error: [%u,%u] expected=%d, actual=%d\n",
                    r, c, h_ref[i], h_C[i]);
      }
      ++errors;
    }
  }

  std::cout << "cleanup\n";
  cleanup();

  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!\n";
    return errors;
  }
  std::cout << "PASSED!\n";
  return 0;
}

