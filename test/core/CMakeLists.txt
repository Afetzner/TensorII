# test/core/CMakeLists.txt

set(SOURCES Tensor_test.cpp)
target_sources(Tests PUBLIC ${SOURCES})

add_subdirectory(ispc)
