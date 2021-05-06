#include "utils.h"

inline double sigmoid(const double& x){
    return 1.0 / (1.0 + std::exp(-x));
}

inline double sigmoid_derivative(const double& x){
    double sigm = sigmoid(x);
    return sigm*(1-sigm);
}

inline double hyp_tan(const double& x){
    return tanh(x);
}

inline double hyp_tan_derivative(const double& x){
    return 1 - tanh(x) * tanh(x);
}

inline double ReLU(const double& x){
    if (x <= 0) return 0;
    return x;
}

inline double ReLU_derivative(const double& x){
    if (x <= 0) return 0;
    return 1;
}

matrix operator*(const matrix& m, double x){
    matrix nm(m.size());
    for (int i = 0; i < m.size(); ++i){
        nm[i].resize(m[i].size());
        for (int j = 0; j < m[i].size(); ++j)
            nm[i][j] = m[i][j]*x;
    }
    return nm;
}

matrix operator*(const matrix& A, const matrix& B){
    int n = A.size();
    int m = B.size();
    int k = B[0].size();
    matrix C(n, Layer(k, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j)
            for (int x = 0; x < m; ++x)
                C[i][j] += A[i][x]*B[x][j];
    return C;
}

Layer operator*(const Layer& a, const matrix& B){
    int n = B.size();
    int m = B[0].size();
    Layer C(m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[j] += a[i]*B[i][j];
    return C;
}

Layer operator*(const Layer& a, const Layer& b){
    int n = a.size();
    Layer c(n);
    for (int i = 0; i < n; ++i)
        c[i] = a[i] * b[i];
    return c;
}

matrix operator+(const matrix& A, const matrix& B){
    int n = A.size();
    int m = A[0].size();
    matrix C(n, Layer(m));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Layer operator*(const matrix& A, const Layer& b){
    int n = A.size();
    int m = b.size();
    Layer c(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            c[i] += A[i][j] * b[j];
    return c;
}

Layer operator+(const Layer& a, const Layer& b){
    int n = a.size();
    Layer c(n);
    for (int i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
    return c;
}

matrix operator!(const matrix& A){
    int n = A.size();
    int m = A[0].size();
    matrix B(m, Layer(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            B[j][i] = A[i][j];
    return B;
}

matrix operator!(const Layer& a){
    int n = a.size();
    matrix B(1, Layer(n));
    for (int i = 0; i < n; ++i)
        B[0][i] = a[i];
    return B;
}

Layer operator-(Layer& a){
    int n = a.size();
    Layer na(n);
    for (int i = 0; i < n; ++i)
        na[i] = -a[i];
    return na;
}

matrix operator-(matrix& A){
    int n = A.size();
    int m = A[0].size();
    matrix B(n, Layer(m));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            B[i][j] = -A[i][j];
    return B;
}

Layer activation_function(const Layer& a){
    int n = a.size();
    Layer b(n);
    for (int i = 0; i < n; ++i)
        b[i] = sigmoid(a[i]);
    return b;
}

Layer activation_function_derivative(const Layer& a){
    int n = a.size();
    Layer b(n);
    for (int i = 0; i < n; ++i)
        b[i] = sigmoid_derivative(a[i]);
    return b;
}

matrixList operator*(const matrixList& m, double x){
    matrixList nm(m.size());
    for (int i = 0; i < m.size(); ++i) {
        nm[i].resize(m[i].size());
        for (int j = 0; j < m[i].size(); ++j) {
            nm[i][j].resize(m[i][j].size());
            for (int k = 0; k < m[i][j].size(); ++k)
                nm[i][j][k] = m[i][j][k] * x;
        }
    }
    return nm;
}

matrixList operator+(const matrixList& A, const matrixList& B){
    matrixList C(A.size());
    for (int i = 0; i < A.size(); ++i) {
        C[i].resize(A[i].size());
        for (int j = 0; j < A[i].size(); ++j) {
            C[i][j].resize(A[i][j].size());
            for (int k = 0; k < A[i][j].size(); ++k)
                C[i][j][k] = A[i][j][k] + B[i][j][k];
        }
    }
    return C;
}

matrixList operator-(const matrixList& A, const matrixList& B){
    matrixList C(A.size());
    for (int i = 0; i < A.size(); ++i) {
        C[i].resize(A[i].size());
        for (int j = 0; j < A[i].size(); ++j) {
            C[i][j].resize(A[i][j].size());
            for (int k = 0; k < A[i][j].size(); ++k)
                C[i][j][k] = A[i][j][k] - B[i][j][k];
        }
    }
    return C;
}

matrix operator-(const matrix& A, const matrix& B){
    matrix C(A.size());
    for (int i = 0; i < A.size(); ++i) {
        C[i].resize(A[i].size());
        for (int j = 0; j < A[i].size(); ++j)
            C[i][j] = A[i][j] - B[i][j];
    }
    return C;
}

matrix layer_to_matrix(const Layer& a){
    matrix B(a.size(), Layer(1));
    for (int i = 0; i < a.size(); ++i)
        B[i][0] = a[i];
    return B;
}