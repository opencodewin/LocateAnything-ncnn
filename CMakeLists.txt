cmake_minimum_required(VERSION 3.18)
project(locate-anything-ncnn LANGUAGES C CXX)

# ============================================================================
# LocateAnything-3B standalone runtime on ncnn.
# - ncnn comes in via a git submodule at 3rdparty/ncnn.
# - Only the LocateAnything runtime code is compiled (vendored in src/).
# - OpenGL/Vulkan is optional: the runtime + base-guard switch CPU<->Vulkan with
#   the same export; fp32 vs fp16 is decided by which model dir you load.
# ============================================================================

option(LOCATE_NCNN_VULKAN  "Build ncnn with the Vulkan GPU backend (CPU fallback still works)" ON)
option(LOCATE_STATIC_NCNN  "Build static ncnn library vs shared" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT MSVC)
    add_compile_options(-Wall -Wno-unused-parameter)
endif()

# ---- ncnn (git submodule) ----
set(_NCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn")
if(NOT EXISTS "${_NCNN_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "ncnn submodule missing. Run: git submodule update --init --recursive")
endif()

# ---- apply vendored patches to the ncnn submodule (idempotent) ----
# Each *.patch in patches/ is applied once at configure time, before ncnn is
# built. Whether it is already applied is detected by a reverse apply check.
find_package(Git REQUIRED)
set(_NCNN_PATCHES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/patches")
file(GLOB _NCNN_PATCHES "${_NCNN_PATCHES_DIR}/*.patch")
# file(GLOB) does not guarantee ordering; apply in the documented
# numerical-prefix order so later patches can build on earlier ones.
list(SORT _NCNN_PATCHES)
foreach(_p ${_NCNN_PATCHES})
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${_NCNN_DIR}" apply --reverse --check "${_p}"
        RESULT_VARIABLE _rev_result
        OUTPUT_QUIET ERROR_QUIET)
    if(_rev_result EQUAL 0)
        message(STATUS "ncnn patch already applied: ${_p}")
        continue()
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${_NCNN_DIR}" apply --check "${_p}"
        RESULT_VARIABLE _chk_result
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT _chk_result EQUAL 0)
        message(FATAL_ERROR
            "ncnn patch does not apply cleanly (${_p}). "
            "The submodule source does not match the patch base. "
            "Run: git -C ${_NCNN_DIR} submodule update --init --recursive, or adjust patches/${_p}.")
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${_NCNN_DIR}" apply "${_p}"
        RESULT_VARIABLE _apply_result
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT _apply_result EQUAL 0)
        message(FATAL_ERROR "failed to apply ncnn patch: ${_p}")
    endif()
    message(STATUS "applied ncnn patch: ${_p} (basename: ${_p})")
endforeach()
unset(_NCNN_PATCHES_DIR)

set(NCNN_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON          OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN          ${LOCATE_NCNN_VULKAN} CACHE BOOL "" FORCE)
if(LOCATE_STATIC_NCNN)
    set(NCNN_SHARED OFF CACHE BOOL "" FORCE)
else()
    set(NCNN_SHARED ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(${_NCNN_DIR} ${CMAKE_BINARY_DIR}/_ncnn_build)

if(TARGET ncnn)
    set(_NCNN_TARGET ncnn)
else()
    message(FATAL_ERROR "ncnn target not found after add_subdirectory")
endif()

# ---- vendored nlohmann/json (header-only, single include at src/nlohmann) ----
add_library(nlohmann_json_iface INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json_iface)
target_include_directories(nlohmann_json_iface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- LocateAnything runtime (vendored minimal code set) ----
add_library(locate_anything STATIC
    src/ncnn_llm_locateanything.cpp
    src/utils/image_utils.cpp
    src/utils/draw_utils.cpp
    src/utils/tokenizer/bpe_tokenizer.cpp
)
target_include_directories(locate_anything PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(locate_anything PUBLIC ${_NCNN_TARGET} nlohmann_json::nlohmann_json)
set_target_properties(locate_anything PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WIN32 OR MINGW)
    target_link_libraries(locate_anything PUBLIC user32 gdi32)
    if(MINGW)
        target_link_libraries(locate_anything PUBLIC ws2_32)
    endif()
endif()

# Threads detection is genuinely cross-platform: macOS and Linux need pthread,
# MSVC links the correct CRT internally but Threads::Threads resolves to the
# right library everywhere, so no platform guard should exclude any platform.
find_package(Threads REQUIRED)
target_link_libraries(locate_anything PUBLIC Threads::Threads)

# ---- CLI entry ----
add_executable(locate_main src/locate_main.cpp)
target_link_libraries(locate_main PRIVATE locate_anything)
set_target_properties(locate_main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# ---- cross-platform bench: cpu-fp32 / gpu-fp32 / gpu-fp16 with CPU-fp32 reference ----
add_executable(bench_platform src/bench_platform.cpp)
target_link_libraries(bench_platform PRIVATE locate_anything)
set_target_properties(bench_platform PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

