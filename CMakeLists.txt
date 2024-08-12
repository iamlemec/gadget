cmake_minimum_required(VERSION 3.21)

project(gadget)
set(BUILD_SHARED_LIBS "ON")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/gadget/libs")
add_subdirectory(llama.cpp)
install(TARGETS llama)
