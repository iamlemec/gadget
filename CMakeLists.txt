cmake_minimum_required(VERSION 3.21)

project(gadget)

set(BUILD_SHARED_LIBS "On")
add_subdirectory(llama.cpp)
install(TARGETS llama)
