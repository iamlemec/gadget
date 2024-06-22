cmake_minimum_required(VERSION 3.21)

project(gadget)

set(BUILD_SHARED_LIBS "On")
add_subdirectory(llama.cpp)
install(TARGETS llama)

set(TARGET inspect)
add_executable(${TARGET} inspect.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
