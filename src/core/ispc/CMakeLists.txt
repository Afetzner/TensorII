# src/core/ispc/CMakeLists.txt

# Use ispc compiler instead of cxx
include(sources.cmake)
add_library(IspcLib STATIC ${SOURCES})
set(CMAKE_ISPC_FLAGS "--target=avx512skx-x16")

target_include_directories(IspcLib
        PRIVATE
        ${Boost_INCLUDE_DIRS}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
#target_link_libraries(IspcLib PRIVATE Boost::headers)
