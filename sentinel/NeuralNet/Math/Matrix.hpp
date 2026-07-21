#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

class Matrix {
public:
    std::vector<std::vector<float>> data;

    Matrix() = default;
    explicit Matrix(std::vector<std::vector<float>> values);

    static Matrix multiply(const Matrix& a, const Matrix& b);
    static Matrix transpose(const Matrix& a);
    static Matrix add(const Matrix& a, const Matrix& b);
    static Matrix subtract(const Matrix& a, const Matrix& b);
    static Matrix scale(const Matrix& a, float s);
    static Matrix multiplyElementwise(const Matrix& a, const Matrix& b);
};

#endif // MATRIX_HPP