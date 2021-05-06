#pragma once

#include "utils.h"

class NNet{
public:
    NNet() = default;
    NNet(const std::vector<int>& topology);
    void fit(const matrix& X, const Layer& y, int iterations, int shot = 2);
    int predict(Layer X);
    double cost(Layer X, int y);

    void train(const std::vector<int>& cur_shuffler, const matrix& X, const Layer& y);

    void propagate_front(int layer);
    Layer propagate_back(int layer, const Layer& delta, matrixList& derivative, matrix& bias_derivative);
private:
    double alpha = 0.3;
    clock_t start;
    std::vector<int> topology;
    matrix mesh, meshZ, bias, bare_bias;
    matrixList weights, bare_weights;
};