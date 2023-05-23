cmake_minimum_required(VERSION 3.0)
project(Neural_network)

set(VAR Neural_network)
set(SOURCES src/main.cpp)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

message("${VAR} is building")

add_executable(${VAR} ${SOURCES})
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_subdirectory(NeuralNetwork)
#add_subdirectory(src)
target_link_libraries(${VAR} NeuralNetwork)