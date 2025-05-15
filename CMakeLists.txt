cmake_minimum_required(VERSION 3.30)
# tips:
#   & "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --version
#   > cmake version 3.30.5-msvc23
project(OpenMeshViewer LANGUAGES CXX C)

# ─── 全局编译选项 ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
# tips: https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html#prop_tgt:CXX_STANDARD
# 优先shared库(dll)而不是static库(lib)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D_USE_MATH_DEFINES)          # 为 OpenMesh 启用数学常量
if(MSVC)                                      # MSVC 让 __cplusplus 返回真实值
    add_compile_options(/Zc:__cplusplus)
endif()

# ─── 第三方依赖路径 ───────────────────────────────────────────
# 设置路径变量默认值，如果在CMakePresets.json中未设置
if(NOT DEFINED QT_DIR)
    set(QT_DIR "C:/Qt/6.9.0/msvc2022_64/lib/cmake" CACHE PATH "Qt cmake directory")
endif()

if(NOT DEFINED OPENMESH_DIR)
    set(OPENMESH_DIR "C:/Program Files/OpenMesh 11.0" CACHE PATH "OpenMesh installation directory")
endif()

# Qt
set(CMAKE_PREFIX_PATH ${QT_DIR})

find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets Gui)

# OpenMesh
set(OpenMesh_DIR "${OPENMESH_DIR}/share/OpenMesh/cmake")
find_package(OpenMesh REQUIRED)
link_directories("${OPENMESH_DIR}/lib")  # 供导入库查找
set(OPENMESH_BIN "${OPENMESH_DIR}")  # 添加 OpenMesh 二进制文件路径变量

# Eigen (header-only library)
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# ─── 子工程 ──────────────────────────────────────────────────
add_subdirectory(src)
