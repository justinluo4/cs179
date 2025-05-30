cmake_minimum_required(VERSION 3.24)  # for CUDA architecture native
project(blackhole LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

find_package(PkgConfig REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPU_DEMO_FILES
        src/cpu_demo.cpp
        src/file_io.cpp
)
set(GPU_DEMO_FILES
        src/gpu_demo.cu
        src/file_io.cpp
        src/HostBuffer.cpp
        src/CudaBuffer.cu
)

add_executable(gpu-demo ${GPU_DEMO_FILES})
add_executable(cpu-demo ${CPU_DEMO_FILES})

