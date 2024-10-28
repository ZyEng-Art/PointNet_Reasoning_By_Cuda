cmake_minimum_required(VERSION 3.10)
project(CUDA_TEST LANGUAGES CUDA CXX)

# 手动指定 HDF5 C++ 库路径
set(HDF5_CXX_LIBRARIES "/usr/local/lib/libhdf5_cpp.so")

# 查找 HDF5 库（包含 C++ 库）
find_package(HDF5 REQUIRED)

# 检查是否支持 CUDA
include(CheckLanguage)
check_language(CUDA)

# 输出找到的 HDF5 路径信息
message(STATUS "HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")
message(STATUS "HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}")

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70)

# 添加编译选项
add_compile_options(-Xcompiler "-std=c++14 -g -O0")

file(GLOB CUDA_SOURCES src/*.cu)
# 创建可执行文件
add_executable(main ${CUDA_SOURCES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 链接 HDF5 和 HDF5 C++ 库
target_link_libraries(main ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})
