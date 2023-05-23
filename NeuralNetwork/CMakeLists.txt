add_library(NeuralNetwork SHARED NeuralNetwork.cpp)
target_include_directories(NeuralNetwork PUBLIC include)
target_link_libraries (NeuralNetwork Eigen3::Eigen)