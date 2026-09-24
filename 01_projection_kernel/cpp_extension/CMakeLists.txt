cmake_minimum_required(VERSION 3.27)
project(mlx_cpp_projection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mlx; print(mlx.__path__[0])"
  OUTPUT_VARIABLE MLX_PACKAGE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(MLX CONFIG REQUIRED
  PATHS "${MLX_PACKAGE}/share/cmake/MLX" NO_DEFAULT_PATH)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED PATHS "${NANOBIND_CMAKE_DIR}" NO_DEFAULT_PATH)

nanobind_add_module(_ext NB_STATIC STABLE_ABI NB_DOMAIN mlx
  src/extension.cpp)
target_link_libraries(_ext PRIVATE mlx)
target_link_options(_ext PRIVATE "-Wl,-rpath,@loader_path/../mlx/lib")
