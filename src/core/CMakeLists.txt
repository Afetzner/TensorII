# src/core/CMakeLists.txt

add_library(CoreLib STATIC)

set(Boost_USE_STATIC_LIBS ON)
#find_package(Boost 1.82 REQUIRED COMPONENTS log)
#target_link_libraries(CoreLib PRIVATE Boost::boost)

# Add core code
include(sources.cmake)
target_sources(CoreLib PUBLIC ${SOURCES})

# Add ISPC dependent code
add_subdirectory(ispc)
target_link_libraries(CoreLib PUBLIC IspcLib)

target_include_directories(CoreLib
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

if(NO_FRAME_POINTER_OPTIMIZATION)
    target_link_options(CoreLib PUBLIC /Oy-)
endif ()
