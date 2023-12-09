cmake_minimum_required(VERSION 3.25)

project(Tensor CXX ISPC)

set(CMAKE_CXX_STANDARD 20)

option(CLEAN_FIRST ON)

include(CMakePrintHelpers)

add_subdirectory(cmake)
add_subdirectory(src)
add_subdirectory(test)
