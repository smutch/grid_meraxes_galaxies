cmake_minimum_required(VERSION 3.2)
project(grid_meraxes_galaxies)

# output paths
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# write compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# compile flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")


set(SRC
    src/grid_meraxes_galaxies.cpp
    )

add_executable(grid_meraxes_galaxies ${SRC})
set_property(TARGET grid_meraxes_galaxies PROPERTY CXX_STANDARD 20)

# 3rd party libs
add_subdirectory(vendor/fmt EXCLUDE_FROM_ALL)
target_link_libraries(grid_meraxes_galaxies PRIVATE fmt::fmt)

find_package(OpenMP)
target_link_libraries(grid_meraxes_galaxies PRIVATE OpenMP::OpenMP_CXX)

find_package(HDF5 REQUIRED COMPONENTS CXX)
target_link_libraries(grid_meraxes_galaxies PRIVATE ${HDF5_CXX_LIBRARIES})
target_include_directories(grid_meraxes_galaxies PRIVATE ${HDF5_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options)
target_link_libraries(grid_meraxes_galaxies PRIVATE fmt Boost::program_options)
