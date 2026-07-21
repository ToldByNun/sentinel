#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

class Matrix {
public:
    std::vector<std::vector<float>> data;

    Matrix() = default;
    explicit Matrix(std::vector<std::vector<float>> values);

    static Matrix multiply(const Matrix& left, const Matrix& right);
    static Matrix transpose(const Matrix& matrix);
    static Matrix add(const Matrix& left, const Matrix& right);
    static Matrix subtract(const Matrix& left, const Matrix& right);
    static Matrix scale(const Matrix& matrix, float scalar);
    static Matrix multiplyElementwise(const Matrix& left, const Matrix& right);
};

#endif // MATRIX_HPP
