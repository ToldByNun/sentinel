#include "ReLU.hpp"

Matrix ReLU::apply(Matrix a) {
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            a.data[i][j] = a.data[i][j] > 0.0f ? a.data[i][j] : 0.0f;
        }
    }
    return a;
}

Matrix ReLU::derivative(Matrix a) {
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            a.data[i][j] = a.data[i][j] > 0.0f ? 1.0f : 0.0f;
        }
    }
    return a;
}
