cmake_minimum_required(VERSION 3.10)

# Project name
project(PossionSolver)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(muparser REQUIRED)

# Add library
add_library(solver SHARED
    src/solver.cpp
)

# Add executable
add_executable(${PROJECT_NAME}
    src/main.cpp
)

# Include directories
target_include_directories(solver
    PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/hdr
)


target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/hdr
)

# Link libraries
target_link_libraries(solver
    PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    muparser
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    solver
)

# Set output directory for binary and library
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin
)

set_target_properties(solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib
)
