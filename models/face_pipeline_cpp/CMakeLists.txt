# CMakeLists.txt for Face Pipeline C++ Backend
cmake_minimum_required(VERSION 3.18)
project(triton_face_pipeline LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ensure symbols are exported properly for dynamic loading
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)

# Find required packages
find_package(CUDA REQUIRED)

# Triton Backend headers
set(TRITON_BACKEND_REPO $ENV{TRITON_BACKEND_REPO})
if(NOT TRITON_BACKEND_REPO)
    set(TRITON_BACKEND_REPO "/opt/triton-backend")
endif()

set(TRITON_COMMON_REPO $ENV{TRITON_COMMON_REPO})
if(NOT TRITON_COMMON_REPO)
    set(TRITON_COMMON_REPO "/opt/triton-common")
endif()

set(TRITON_CORE_REPO $ENV{TRITON_CORE_REPO})
if(NOT TRITON_CORE_REPO)
    set(TRITON_CORE_REPO "/opt/triton-core")
endif()

# Include directories
include_directories(
    ${TRITON_BACKEND_REPO}/include
    ${TRITON_COMMON_REPO}/include
    ${TRITON_CORE_REPO}/include
    ${CUDA_INCLUDE_DIRS}
    /usr/include/rapidjson
)

# Find TensorRT
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/x86_64-linux-gnu)

# Triton common utility sources
set(TRITON_COMMON_SOURCES
    ${TRITON_COMMON_REPO}/src/async_work_queue.cc
    ${TRITON_COMMON_REPO}/src/error.cc
    ${TRITON_COMMON_REPO}/src/logging.cc
    ${TRITON_COMMON_REPO}/src/thread_pool.cc
)

# Triton backend utility sources (must be compiled in)
set(TRITON_BACKEND_SOURCES
    ${TRITON_BACKEND_REPO}/src/backend_common.cc
    ${TRITON_BACKEND_REPO}/src/backend_input_collector.cc
    ${TRITON_BACKEND_REPO}/src/backend_memory.cc
    ${TRITON_BACKEND_REPO}/src/backend_model.cc
    ${TRITON_BACKEND_REPO}/src/backend_model_instance.cc
    ${TRITON_BACKEND_REPO}/src/backend_output_responder.cc
    ${TRITON_BACKEND_REPO}/src/kernel.cu
    ${TRITON_COMMON_SOURCES}
)

# Source files
set(SOURCES
    src/face_pipeline.cc
    src/gpu_crop.cu
    ${TRITON_BACKEND_SOURCES}
)

# Build shared library
add_library(triton_face_pipeline SHARED ${SOURCES})

target_link_libraries(triton_face_pipeline
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

set_target_properties(triton_face_pipeline PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""  # No 'lib' prefix
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

# Add compile definitions for Triton backend
target_compile_definitions(triton_face_pipeline PRIVATE
    TRITON_ENABLE_GPU=1
)

# Install
install(TARGETS triton_face_pipeline
    LIBRARY DESTINATION backends/face_pipeline
)
