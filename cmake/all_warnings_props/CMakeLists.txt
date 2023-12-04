# cmake/warning_props/CMakeLists.txt
add_library(all_warning_props INTERFACE)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    message("Using MSVC warning level 4")
    add_compile_options(/W4)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    message("Using all GCC warning flags")
    target_compile_options(all_warning_props INTERFACE
            -pedantic
            -Wall
            -Wextra
            -Wcast-align
            -Wcast-qual
            -Wctor-dtor-privacy
            -Wdisabled-optimization
            -Wformat=2
            -Winit-self
            -Wlogical-op
            -Wmissing-declarations
            -Wmissing-include-dirs
            -Wnoexcept
            -Wold-style-cast
            -Woverloaded-virtual
            -Wredundant-decls
            -Wshadow
            -Wsign-conversion
            -Wsign-promo
            -Wstrict-null-sentinel
            -Wstrict-overflow=5
            -Wswitch-default
            -Wundef
            -Werror
            -Wno-unused)
endif ()