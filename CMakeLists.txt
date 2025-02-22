cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MLP_examples)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the compile commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add external/glm to the CMake subfolder
add_subdirectory(external/glm)

# Add external/glm to the include folders
include_directories(external/glm)
include_directories(external/)
include_directories(src/)

# Add the executable
add_executable(TrainModel train.cpp
                         src/MLP.cpp)

add_executable(ReconstructImage reconstruct.cpp
                                src/MLP.cpp)
