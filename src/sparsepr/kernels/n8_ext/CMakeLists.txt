cmake_minimum_required(VERSION 3.26.4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
endif()
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
  set(TORCH_CUDA_ARCH_LIST "8.0;8.6;9.0" CACHE STRING "PyTorch CUDA architectures")
endif()

project(n8_kernels LANGUAGES CUDA CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

if(NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "torch_python was not found below ${TORCH_INSTALL_PREFIX}/lib")
endif()

set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use CMake FindPython for pybind11" FORCE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(n8_kernels MODULE csrc/n8_ops.cu)
target_compile_options(
  n8_kernels PRIVATE
  ${TORCH_CXX_FLAGS_LIST}
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-extended-lambda
    --expt-relaxed-constexpr
    --use_fast_math
    --disable-warnings
  >
  $<$<COMPILE_LANGUAGE:CXX>:-w>
)
target_link_libraries(
  n8_kernels PRIVATE
  ${TORCH_LIBRARIES}
  Python::Python
  pybind11::module
  ${TORCH_PYTHON_LIBRARY}
)
