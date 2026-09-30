cmake_minimum_required(VERSION 3.24)
project(palladium LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compile_commands.json for clangd, also when built without scikit-build-core.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# build-dir = "build/{wheel_tag}" varies per ABI, out of clangd's search
# path, so symlink the file to the source root.
add_custom_target(
  symlink_compile_commands ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_BINARY_DIR}/compile_commands.json
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  COMMENT "Symlinking compile_commands.json to source root"
  VERBATIM)

# find_package(Python) searches PATH, which may miss the project venv;
# scikit-build-core sets $VIRTUAL_ENV. Override with -DPython_EXECUTABLE.
if(NOT DEFINED Python_EXECUTABLE AND DEFINED ENV{VIRTUAL_ENV})
  set(_venv_python "$ENV{VIRTUAL_ENV}/bin/python3")
  if(EXISTS "${_venv_python}")
    set(Python_EXECUTABLE "${_venv_python}")
  endif()
endif()
if(NOT DEFINED Python_EXECUTABLE)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jax; print(jax.ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "from metal_runtime import c_api; print(c_api.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE METAL_RUNTIME_INCLUDE_DIR
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "from metal_runtime import c_api; print(c_api.library_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE METAL_RUNTIME_LIBRARY_DIR
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "jax.ffi.include_dir(): ${JAX_FFI_INCLUDE_DIR}")
message(STATUS "metal_runtime.c_api.include_dir(): ${METAL_RUNTIME_INCLUDE_DIR}")
message(STATUS "metal_runtime.c_api.library_dir(): ${METAL_RUNTIME_LIBRARY_DIR}")

add_library(palladium_ffi SHARED native/ffi/palladium_ffi.cpp)
target_include_directories(palladium_ffi PRIVATE
  "${JAX_FFI_INCLUDE_DIR}"
  "${METAL_RUNTIME_INCLUDE_DIR}"
)
find_library(METAL_RUNTIME_C_LIB metal_runtime PATHS "${METAL_RUNTIME_LIBRARY_DIR}" REQUIRED)
target_link_libraries(palladium_ffi PRIVATE "${METAL_RUNTIME_C_LIB}")

# The absolute rpath serves editable builds (dylib in the CMake build dir);
# @loader_path serves wheel installs, where both packages are siblings in
# site-packages.
set_target_properties(palladium_ffi PROPERTIES
  INSTALL_RPATH "${METAL_RUNTIME_LIBRARY_DIR};@loader_path/../metal_runtime"
  BUILD_WITH_INSTALL_RPATH ON
)

# Found at runtime through importlib.resources (ffi._library_path).
install(TARGETS palladium_ffi LIBRARY DESTINATION palladium)
