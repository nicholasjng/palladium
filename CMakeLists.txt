cmake_minimum_required(VERSION 3.24)
project(palladium LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Emit compile_commands.json for clangd / IDE LSPs. Also force-set here
# in case the project is built without scikit-build-core (e.g. raw
# `cmake -B`).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# The file lands in ${CMAKE_BINARY_DIR}; with build-dir = "build/{wheel_tag}"
# in pyproject.toml that path varies per ABI, so clangd's auto-discovery
# (which probes ./build/ and the source root) won't find it. Symlink it to
# the source root so IDE tooling picks it up without configuration.
add_custom_target(
  symlink_compile_commands ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_BINARY_DIR}/compile_commands.json
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  COMMENT "Symlinking compile_commands.json to source root"
  VERBATIM)

# find_package(Python) prefers PATH, not reliably the project venv.
# Prefer $VIRTUAL_ENV/bin/python3 when set (scikit-build-core sets it to
# the venv it's building for); override with -DPython_EXECUTABLE=...
# otherwise.
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

# The jax.ffi entry point (native/ffi/palladium_ffi.cpp): one generic
# handler bridging metal-runtime's C API to XLA FFI. See that file's own
# header comment for what it does.
add_library(palladium_ffi SHARED native/ffi/palladium_ffi.cpp)
target_include_directories(palladium_ffi PRIVATE
  "${JAX_FFI_INCLUDE_DIR}"
  "${METAL_RUNTIME_INCLUDE_DIR}"
)
find_library(METAL_RUNTIME_C_LIB metal_runtime_c PATHS "${METAL_RUNTIME_LIBRARY_DIR}" REQUIRED)
target_link_libraries(palladium_ffi PRIVATE "${METAL_RUNTIME_C_LIB}")

# Two rpath entries, both needed: the absolute one resolves in an
# editable/dev build, where palladium_ffi.dylib lives in the CMake build
# dir with no fixed relationship to metal_runtime's install location;
# the @loader_path one resolves in a real wheel install, where both
# packages land as siblings under some site-packages whose absolute path
# isn't known at build time. dyld tries each until one works.
set_target_properties(palladium_ffi PROPERTIES
  INSTALL_RPATH "${METAL_RUNTIME_LIBRARY_DIR};@loader_path/../metal_runtime"
  BUILD_WITH_INSTALL_RPATH ON
)

# Ships inside the installed palladium package; see
# src/palladium/ffi.py's _library_path() for how it's found at runtime
# (the same importlib.resources pattern metal_runtime.c_api uses).
install(TARGETS palladium_ffi LIBRARY DESTINATION palladium)
