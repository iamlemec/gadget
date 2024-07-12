cmake_minimum_required(VERSION 3.21)

project(gadget)
set(BUILD_SHARED_LIBS "On")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/gadget")
add_subdirectory(llama.cpp)
install(TARGETS llama)
