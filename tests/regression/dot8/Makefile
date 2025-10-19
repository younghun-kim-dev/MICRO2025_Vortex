ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# 프로젝트 이름 갱신
PROJECT := dot8

# 이 테스트의 소스 루트
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

# 호스트 실행 바이너리
SRCS := $(SRC_DIR)/main.cpp

# 디바이스 커널(vxbin 생성)
VX_SRCS := $(SRC_DIR)/kernel.cpp

# 기본 옵션 (원한다면 N 조절)
OPTS ?= -n32

