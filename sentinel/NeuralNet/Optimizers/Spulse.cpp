#include "Spulse.hpp"

#include <stdexcept>

SpulseState SpulseState::zerosLike(const Matrix& parameter) {
    SpulseState state;
    state.ensure(parameter);
    return state;
}

void SpulseState::ensure(const Matrix& parameter) {
    if (parameter.empty()) throw std::invalid_argument("SpulseState::ensure empty parameter");
    if (this->momentum.rows == parameter.rows && this->momentum.cols == parameter.cols
        && this->momentum.data.size() == parameter.data.size())
        return;
    this->momentum.ensureSize(parameter.rows, parameter.cols);
    this->momentum.fill(0.0f);
    this->energyFast = 0.0f;
    this->energySlow = 0.0f;
}

void SpulseState::clear() {
    this->momentum = Matrix();
    this->energyFast = 0.0f;
    this->energySlow = 0.0f;
}
