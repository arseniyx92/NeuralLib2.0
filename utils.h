#pragma once


// INCLUDES
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <random>

// DEFINES

using Layer = std::vector<double>;
using matrix = std::vector<Layer>;
using matrixList = std::vector<matrix>;

// FUNCTIONS

inline double sigmoid(const double& x);

inline double sigmoid_derivative(const double& x);

inline double hyp_tan(const double& x);

inline double hyp_tan_derivative(const double& x);

inline double ReLU(const double& x);

inline double ReLU_derivative(const double& x);

matrix operator*(const matrix& m, double x);

matrix operator*(const matrix& A, const matrix& B);

Layer operator*(const Layer& a, const matrix& B);

Layer operator*(const Layer& a, const Layer& b);

matrix operator+(const matrix& A, const matrix& B);

Layer operator*(const matrix& A, const Layer& b);

Layer operator+(const Layer& a, const Layer& b);

matrix operator!(const matrix& A);

matrix operator!(const Layer& a);

Layer operator-(Layer& a);

matrix operator-(matrix& A);

Layer activation_function(const Layer& a);

Layer activation_function_derivative(const Layer& a);

matrixList operator*(const matrixList& m, double x);

matrixList operator-(const matrixList& A, const matrixList& B);

matrix operator-(const matrix& A, const matrix& B);

matrixList operator+(const matrixList& A, const matrixList& B);

matrix operator+(const matrix& A, const matrix& B);

matrix layer_to_matrix(const Layer& a);