cmake_minimum_required(VERSION 3.21)

# create project
project(gadget)
set(BUILD_SHARED_LIBS "ON")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/gadget/libs")

# add debug symbols
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# add subdirectory
add_subdirectory(ggml)
install(TARGETS ggml)
