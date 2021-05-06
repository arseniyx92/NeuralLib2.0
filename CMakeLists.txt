cmake_minimum_required(VERSION 3.17)
project(NeuralNetwork2_0)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetwork2_0 main.cpp neural_net.cpp neural_net.h utils.h utils.cpp)