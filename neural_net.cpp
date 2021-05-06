#include "neural_net.h"


NNet::NNet(const std::vector<int>& _topology){
    topology = _topology;
    int n = topology.size();
    mesh.resize(n);
    bias.resize(n);
    weights.resize(n);
    for (int i = 0; i < n; ++i){
        bias[i].resize(topology[i]);
        mesh[i].resize(topology[i]);
        if (i != n-1) weights[i].resize(topology[i], Layer(topology[i+1]));
    }
    meshZ = mesh;
    bare_bias = bias;
    bare_weights = weights;
}

void NNet::fit(const matrix& X, const Layer& y, int iterations, int shot){
    std::mt19937 rnd(42);
    int n = X.size();
    std::vector<int> shuffler(n);
    for (int i = 0; i < n; ++i)
        shuffler[i] = i;
    shuffle(shuffler.begin(), shuffler.end(), rnd);
    int cnt = 0;
    while (cnt < n) {
        std::vector<int> cur_shuffler;
        if (cnt + shot > n) {
            for (; cnt < n; ++cnt)
                cur_shuffler.push_back(shuffler[cnt]);
        } else {
            for (int i = cnt; i < shot + cnt; ++i)
                cur_shuffler.push_back(shuffler[i]);
            cnt += shot;
        }
        for (int iter = 0; iter < iterations; ++iter)
            train(cur_shuffler, X, y);
    }
}

void NNet::propagate_front(int layer){
    meshZ[layer+1] = ((!weights[layer]) * mesh[layer]) + bias[layer+1];
    mesh[layer+1] = activation_function(meshZ[layer+1]);
}

Layer NNet::propagate_back(int layer, const Layer& delta, matrixList& derivative, matrix& bias_derivative){
    Layer ndelta = weights[layer] * delta;
    ndelta = ndelta * activation_function_derivative(meshZ[layer]);

    bias_derivative[layer] = bias_derivative[layer] + ndelta;
    derivative[layer-1] = derivative[layer-1] + (layer_to_matrix(mesh[layer-1]) * (!delta));
    return ndelta;
}

void NNet::train(const std::vector<int>& cur_shuffler, const matrix& X, const Layer& y) {
    matrixList derivative = bare_weights;
    matrix bias_derivative = bare_bias;
    for (int cur:cur_shuffler) {
        // receiving input
        for (int i = 0; i < topology[0]; ++i)
            mesh[0][i] = X[cur][i];
        meshZ[0] = mesh[0];

        // front propagation
        for (int layer = 0; layer < (int) topology.size() - 1; ++layer)
            propagate_front(layer);

        // generating delta of the last layer
        Layer delta(topology.back(), 0);
        delta[y[cur]] = 1.0;
        delta = mesh.back() + (-delta);
        delta = delta * activation_function(meshZ.back());

        // updating derivative for the last layer
        bias_derivative.back() = bias_derivative.back() + delta;
        derivative[mesh.size() - 2] = derivative[mesh.size() - 2] + (layer_to_matrix(mesh[mesh.size() - 2]) * (!delta));

        // back propagating
        for (int layer = (int) topology.size() - 2; layer >= 1; --layer)
            delta = propagate_back(layer, delta, derivative, bias_derivative);
    }

    derivative = derivative * (1.0 / (double) cur_shuffler.size());
    bias_derivative = bias_derivative * (1.0 / (double) cur_shuffler.size());

    weights = weights - derivative;
    bias = bias - bias_derivative;

    if (clock() - start > 10000){
        start = clock();
        double summ = cost({1, 1}, 0) + cost({1, 0}, 1) + cost({0, 1}, 1) + cost({0, 0}, 0);
        summ /= 4.;
        std::cout << summ << std::endl;
    }
}

int NNet::predict(Layer X){
    // receiving input
    for (int i = 0; i < topology[0]; ++i)
        mesh[0][i] = X[i];

    // front propagation
    for (int layer = 0; layer < (int) topology.size() - 1; ++layer)
        propagate_front(layer);

    // calculating the best variant
    int best_variant = 0;
    for (int i = 0; i < topology.back(); ++i)
        if (mesh.back()[i] > mesh.back()[best_variant]) best_variant = i;
    return best_variant;
}

double NNet::cost(Layer X, int y){

    // generating valid last layer
    Layer Y(topology.back(), 0);
    Y[y] = 1;

    // receiving input
    for (int i = 0; i < topology[0]; ++i)
        mesh[0][i] = X[i];

    // front propagation
    for (int layer = 0; layer < (int) topology.size() - 1; ++layer)
        propagate_front(layer);

    // calculating cost function
    double cost = 0;
    for (int i = 0; i < topology.back(); ++i)
        cost += (Y[i] - mesh.back()[i])*(Y[i] - mesh.back()[i]);
    return cost;
}