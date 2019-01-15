# Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

LOCAL_PATH := $(call my-dir)

ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
   ifeq ($(APP_STL), gnustl_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/aarch64-android-gcc4.9
   else ifeq ($(APP_STL), c++_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/aarch64-android-clang3.8
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
   ifeq ($(APP_STL), gnustl_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/arm-android-gcc4.9
   else ifeq ($(APP_STL), c++_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/arm-android-clang3.8
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else
   $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

SNPE_INCLUDE_DIR := $(SNPE_ROOT)/include/zdl

include $(CLEAR_VARS)
LOCAL_MODULE := snpe-sample
LOCAL_SRC_FILES := main.cpp CheckRuntime.cpp LoadContainer.cpp LoadInputTensor.cpp SetBuilderOptions.cpp Util.cpp NV21Load.cpp udlExample.cpp CreateUserBuffer.cpp PreprocessInput.cpp SaveOutputTensor.cpp CreateGLBuffer.cpp
LOCAL_SHARED_LIBRARIES := libSNPE libSYMPHONYCPU libSYMPHONYPOWER
LOCAL_LDLIBS     := -lGLESv2 -lEGL
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := libSNPE
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libSNPE.so
LOCAL_EXPORT_C_INCLUDES += $(SNPE_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libSYMPHONYCPU
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libsymphony-cpu.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libSYMPHONYPOWER
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libsymphonypower.so
include $(PREBUILT_SHARED_LIBRARY)
