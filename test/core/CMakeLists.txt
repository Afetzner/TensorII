# test/core/CMakeLists.txt

include(tests.cmake)
target_sources(Tests PUBLIC ${SOURCES})

add_subdirectory(ispc)
