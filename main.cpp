#include "utils.h"
#include "neural_net.h"

using namespace std;

int main() {
    mt19937 rnd(time(NULL));
    matrix X;
    Layer y;
//    X = {{1, 1},
//         {1, 0},
//         {0, 1},
//         {0, 0}};
//    y = {0, 1, 1, 0};
//        X = {{1, 1}};
//        y = {0};
        X = {{0, 0}, {1, 0}};
        y = {0, 1};
//    for (int i = 0; i < 200; ++i){
//        X.push_back({(double)(rnd()%2), (double)(rnd()%2)});
//        y.push_back({(double)((int)X.back()[0] ^ (int)X.back()[1])});
//    }
    NNet net({2, 2, 2, 2});
    net.fit(X, y, 20000, 2);
    cout << net.predict({1, 1}) << std::endl;
    cout << net.predict({1, 0}) << std::endl;
    cout << net.predict({0, 1}) << std::endl;
    cout << net.predict({0, 0}) << std::endl;
    return 0;
}