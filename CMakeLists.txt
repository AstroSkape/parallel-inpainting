cmake_minimum_required(VERSION 3.10)
project(ParallelInpainting)

set(CMAKE_CXX_STANDARD 17)

if(APPLE)
    set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
elseif(UNIX)
    set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(inpaint src/main.cpp src/inpaint.cpp src/masked_image.cpp src/nnf.cpp)

target_link_libraries(inpaint ${OpenCV_LIBS})