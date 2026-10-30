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
message(STATUS "jax.ffi.include_dir(): ${JAX_FFI_INCLUDE_DIR}")

# Apple's metal-cpp, header-only. Some releases nest under metal-cpp/.
include(FetchContent)
FetchContent_Declare(
  metalcpp
  URL https://developer.apple.com/metal/cpp/files/metal-cpp_26.4.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(metalcpp)
if(EXISTS "${metalcpp_SOURCE_DIR}/Metal/Metal.hpp")
  set(METAL_CPP_INCLUDE "${metalcpp_SOURCE_DIR}")
else()
  set(METAL_CPP_INCLUDE "${metalcpp_SOURCE_DIR}/metal-cpp")
endif()
add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE "${METAL_CPP_INCLUDE}")
target_compile_definitions(metal-cpp INTERFACE METALCPP_SYMBOL_VISIBILITY_HIDDEN)

# The Metal runtime behind the C API in native/runtime/c_api.h: device, queue,
# library cache and dispatch. Static, so whatever links it carries its own
# copy; the FFI handler exports the mr_* functions for ctypes callers.
add_library(palladium_runtime STATIC
  native/runtime/runtime.cpp
  native/runtime/metal.cpp
  native/runtime/buffer.cpp
  native/runtime/library.cpp
  native/runtime/dispatch.cpp
  native/runtime/c_api.cpp)
add_library(palladium::runtime ALIAS palladium_runtime)
target_include_directories(palladium_runtime PUBLIC native/runtime)
target_link_libraries(palladium_runtime
  PUBLIC metal-cpp "-framework Foundation" "-framework Metal")
set_target_properties(palladium_runtime PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

add_library(palladium_ffi SHARED native/ffi/palladium_ffi.cpp)
target_include_directories(palladium_ffi PRIVATE "${JAX_FFI_INCLUDE_DIR}")
target_link_libraries(palladium_ffi PRIVATE palladium::runtime)

# The PJRT C API header at the XLA commit jaxlib 0.12 pins (jax's
# MODULE.bazel), so the plugin reports the API version jaxlib expects.
set(XLA_COMMIT d867969800d4a8fd045505eabc65e1e66f30794f)
set(PJRT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/pjrt-include")
file(DOWNLOAD
  "https://raw.githubusercontent.com/openxla/xla/${XLA_COMMIT}/xla/pjrt/c/pjrt_c_api.h"
  "${PJRT_INCLUDE_DIR}/xla/pjrt/c/pjrt_c_api.h"
  EXPECTED_HASH SHA256=174b44736ccb200cb402c2466822220971ba578e1e8dfd690c56f822a9e71582)

# The `metal` PJRT plugin, loaded by jax through palladium._pjrt.
add_library(palladium_pjrt SHARED native/pjrt/plugin.cpp)
target_include_directories(palladium_pjrt PRIVATE "${PJRT_INCLUDE_DIR}")
target_link_libraries(palladium_pjrt PRIVATE palladium::runtime)
set_target_properties(palladium_pjrt PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Found at runtime through importlib.resources (ffi._library_path).
install(TARGETS palladium_ffi palladium_pjrt LIBRARY DESTINATION palladium)
