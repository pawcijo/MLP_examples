#include "Perceptron.hpp"
#include <cstdlib>
#include <numeric>

Perceptron::Perceptron(int input_size, float learning_rate)
    : learning_rate(learning_rate), bias(0.0f) {
    weights.resize(input_size);
    for (auto& weight : weights) {
        weight = static_cast<float>(rand()) / RAND_MAX; // Initialize weights randomly
    }
}

void Perceptron::train(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<float> input = inputs[i];
            int label = labels[i];
            float prediction = activation_function(dot_product(input, weights) + bias);
            float error = label - prediction;

            // Update weights and bias
            for (size_t j = 0; j < weights.size(); ++j) {
                weights[j] += learning_rate * error * input[j];
            }
            bias += learning_rate * error;
        }
    }
}

std::vector<float> Perceptron::predict(const std::vector<float>& input) {
    float sum = dot_product(input, weights) + bias;
    return {activation_function(sum)};
}

float Perceptron::activation_function(float x) {
    return x >= 0.0f ? 1.0f : 0.0f;
}

float Perceptron::dot_product(const std::vector<float>& v1, const std::vector<float>& v2) {
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0f);
}