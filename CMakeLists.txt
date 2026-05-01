cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "g++-11")
set(CMAKE_CUDA_HOST_COMPILER "g++-11")

project(ParallelInpainting LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

if(APPLE)
    set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
elseif(UNIX)
    set(OpenCV_DIR "$ENV{HOME}/local/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")
add_executable(inpaint src/app/main.cpp src/patchmatch/inpaint.cpp src/patchmatch/masked_image.cpp src/patchmatch/nnf.cpp src/cuda/nnf_kernels.cu src/cuda/cuda.cpp src/cuda/cuda_helpers.cu)

target_link_libraries(inpaint ${OpenCV_LIBS} OpenMP::OpenMP_CXX CUDA::cudart)
