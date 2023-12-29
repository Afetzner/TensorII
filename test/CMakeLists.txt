# test/CMakeLists.txt

add_executable(Tests test.cpp)

# This is hacky: For some reason I'm not able to find Catch2 when building a releast build
# Use CMAKE_TOOLCHAIN_FILE to get the vcpkg root dir, and then manually specify the catch2 location. Ick
get_filename_component(VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
get_filename_component(VCPKG_ROOT "${VCPKG_ROOT}" DIRECTORY)
set(CMAKE_PREFIX_PATH $<$<CONFIG:Release>:${VCPKG_ROOT}/installed/x86-windows/share/Catch2>)

# Add Catch2
find_package(Catch2 3 REQUIRED)
target_link_libraries(Tests PRIVATE Catch2::Catch2WithMain)

# Link tests to core library
target_link_libraries(Tests PRIVATE CoreLib)

# Link tests exe to core tests library
add_subdirectory(core)

include(CTest)
include(Catch)
catch_discover_tests(Tests)
