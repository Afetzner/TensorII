# test/core/CMakeLists.txt

set(SOURCES foo.cpp)
target_sources(Tests PUBLIC ${SOURCES})

add_subdirectory(ispc)
