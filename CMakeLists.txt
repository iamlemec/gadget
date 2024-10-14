cmake_minimum_required(VERSION 3.21)

# create project
project(gadget)
set(BUILD_SHARED_LIBS "ON")

# set output directory for library
if (SKBUILD GREATER 0)
    set(SOURCE_DIR "${SKBUILD_PLATLIB_DIR}")
else()
    set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SOURCE_DIR}/gadget/libs")

# add debug symbols
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# add subdirectory
add_subdirectory(ggml)
