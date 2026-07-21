#include <iostream>
#include <vector>

struct matrix2D {
    std::vector<std::vector<float>> data;
};

matrix2D multiplyMatrix(const matrix2D& a, const matrix2D& b) {
    size_t rowsA = a.data.size();
    size_t colsA = a.data[0].size();
    size_t colsB = b.data[0].size();

    matrix2D result;
    result.data = std::vector<std::vector<float>>(rowsA, std::vector<float>(colsB, 0.0f));

    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            for (size_t k = 0; k < colsA; k++) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

matrix2D transpose(const matrix2D& a) {
    size_t rows = a.data.size();
    size_t cols = a.data[0].size();

    matrix2D result;
    result.data = std::vector<std::vector<float>>(cols, std::vector<float>(rows, 0.0f));

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[j][i] = a.data[i][j];
        }
    }
    return result;
}

matrix2D add(const matrix2D& a, const matrix2D& b) {
    matrix2D result = a;
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            result.data[i][j] += b.data[i][j];
        }
    }
    return result;
}

matrix2D subtract(const matrix2D& a, const matrix2D& b) {
    matrix2D result = a;
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            result.data[i][j] -= b.data[i][j];
        }
    }
    return result;
}

matrix2D scale(const matrix2D& a, float s) {
    matrix2D result = a;
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            result.data[i][j] *= s;
        }
    }
    return result;
}

matrix2D relu(matrix2D& a) {
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            a.data[i][j] = a.data[i][j] > 0.0f ? a.data[i][j] : 0.0f;
        }
    }
    return a;
}

matrix2D reluDerivative(matrix2D a) {
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            a.data[i][j] = a.data[i][j] > 0.0f ? 1.0f : 0.0f;
        }
    }
    return a;
}

matrix2D multiplyElementwise(const matrix2D& a, const matrix2D& b) {
    matrix2D result = a;
    for (size_t i = 0; i < a.data.size(); ++i) {
        for (size_t j = 0; j < a.data[i].size(); ++j) {
            result.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }
    return result;
}

matrix2D forward(const matrix2D& input, const matrix2D& weight, const matrix2D& bias, matrix2D& z_out) {
    z_out = add(multiplyMatrix(weight, input), bias);
    matrix2D activated = z_out;
    return relu(activated);
}

int main() {
    matrix2D input;
    input.data = { {1.0f}, {2.0f}, {0.5f} };

    matrix2D target;
    target.data = { {2.0f}, {4.0f}, {1.0f} };

    matrix2D weight1;
    weight1.data = {
        {0.1f, 0.2f, 0.1f},
        {0.1f, 0.1f, 0.2f},
        {0.2f, 0.1f, 0.1f},
        {0.1f, 0.1f, 0.1f}
    };
    matrix2D bias1;
    bias1.data = { {0.0f}, {0.0f}, {0.0f}, {0.0f} };

    matrix2D weight2;
    weight2.data = {
        {0.1f, 0.1f, 0.1f, 0.1f},
        {0.1f, 0.2f, 0.1f, 0.1f},
        {0.1f, 0.1f, 0.2f, 0.1f}
    };
    matrix2D bias2;
    bias2.data = { {0.0f}, {0.0f}, {0.0f} };

    for (int epoch = 0; epoch < 10000; ++epoch) {
        matrix2D z1, z2;
        matrix2D hidden = forward(input, weight1, bias1, z1);
        matrix2D prediction = forward(hidden, weight2, bias2, z2);

        matrix2D loss = subtract(prediction, target);

        matrix2D delta2 = multiplyElementwise(loss, reluDerivative(z2));
        matrix2D dW2 = multiplyMatrix(delta2, transpose(hidden));

        matrix2D wt_delta2 = multiplyMatrix(transpose(weight2), delta2);
        matrix2D delta1 = multiplyElementwise(wt_delta2, reluDerivative(z1));
        matrix2D dW1 = multiplyMatrix(delta1, transpose(input));

        weight2 = subtract(weight2, scale(dW2, 0.01f));
        bias2 = subtract(bias2, scale(delta2, 0.01f));

        weight1 = subtract(weight1, scale(dW1, 0.01f));
        bias1 = subtract(bias1, scale(delta1, 0.01f));

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " | Prediction[0]: " << prediction.data[0][0] << '\n';
        }
    }

    return 0;
}