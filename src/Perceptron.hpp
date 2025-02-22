#pragma once

#include <vector>

class Perceptron {
public:
    Perceptron(int input_size, float learning_rate);
    void train(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels, int epochs);
    std::vector<float> predict(const std::vector<float>& input);

private:
    std::vector<float> weights;
    float learning_rate;
    float bias;

    float activation_function(float x);
    float dot_product(const std::vector<float>& v1, const std::vector<float>& v2);
};
