ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := prefetch   # <-- fence -> prefetch 로 교체

# 환경변수 의존 피하려고 ROOT_DIR 기준으로 명시
SRC_DIR := $(ROOT_DIR)/tests/regression/$(PROJECT)

SRCS    := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n64

include ../common.mk
