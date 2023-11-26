# src/core/ispc/CMakeLists.txt

# Use ispc compiler instead of cxx
include(sources.cmake)
add_library(IspcLib STATIC ${SOURCES})
set(CMAKE_ISPC_FLAGS "--target=avx512skx-x16")

target_include_directories(IspcLib
        PRIVATE
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/private>
#        $<INSTALL_INTERFACE:include/private>
        ${Boost_INCLUDE_DIRS}
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/interface>
        $<INSTALL_INTERFACE:include/interface>
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/IspcLib.dir
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/public>
        $<INSTALL_INTERFACE:include/public>
        )
#target_link_libraries(IspcLib PRIVATE Boost::headers)
