#include "MLP.hpp"
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstdlib>  // Include for rand and srand
#include <ctime>    // Include for time

MLP::MLP(const std::vector<int>& layers, float learning_rate) : learning_rate(learning_rate) {
    if (layers.size() < 2) {
        throw std::invalid_argument("There must be at least two layers (input and output).");
    }

    for (int layer_size : layers) {
        if (layer_size <= 0) {
            throw std::invalid_argument("Layer sizes must be greater than zero.");
        }
    }

    // Initialize random seed
    srand(static_cast<unsigned int>(time(nullptr)));

    // Initialize weights and biases
    for (size_t i = 1; i < layers.size(); ++i) {
        weights.push_back(std::vector<std::vector<float>>(layers[i], std::vector<float>(layers[i - 1])));
        for (int j = 0; j < layers[i]; ++j) {
            for (int k = 0; k < layers[i - 1]; ++k) {
                weights[i - 1][j][k] = static_cast<float>(rand()) / RAND_MAX * 2 - 1; // Random weights in [-1, 1]
            }
        }

        biases.push_back(std::vector<float>(layers[i]));
        for (int j = 0; j < layers[i]; ++j) {
            biases[i - 1][j] = static_cast<float>(rand()) / RAND_MAX * 0.1f; // Small random values
        }
    }
}

void MLP::train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& labels, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        for (size_t sample = 0; sample < inputs.size(); ++sample) {
            // Forward pass
            std::vector<std::vector<float>> activations;
            activations.push_back(inputs[sample]);

            for (size_t i = 0; i < weights.size(); ++i) {
                std::vector<float> new_activations(weights[i].size());
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    new_activations[j] = activation_function(dot_product(weights[i][j], activations.back()) + biases[i][j]);
                }
                activations.push_back(new_activations);
            }

            // Backward pass
            std::vector<std::vector<float>> deltas(weights.size());
            for (size_t i = weights.size(); i-- > 0;) {
                deltas[i] = std::vector<float>(weights[i].size());
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    if (i == weights.size() - 1) {
                        deltas[i][j] = (activations.back()[j] - labels[sample][j]) * activations.back()[j] * (1.0f - activations.back()[j]);
                        total_loss += std::pow(deltas[i][j], 2);
                    } else {
                        float sum = 0.0f;
                        for (size_t k = 0; k < weights[i + 1].size(); ++k) {
                            sum += weights[i + 1][k][j] * deltas[i + 1][k];
                        }
                        deltas[i][j] = sum * activations[i + 1][j] * (1.0f - activations[i + 1][j]);
                    }
                }
            }

            // Update weights and biases
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    for (size_t k = 0; k < weights[i][j].size(); ++k) {
                        weights[i][j][k] -= learning_rate * deltas[i][j] * activations[i][k];
                    }
                    biases[i][j] -= learning_rate * deltas[i][j];
                }
            }
        }

        // Show progress in terminal
        if ((epoch + 1) % 100 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs << " - Loss: " << total_loss / inputs.size() << std::endl;
        }
    }
}

std::vector<float> MLP::predict(const std::vector<float>& input) {
    if (input.size() != weights[0][0].size()) { // Assuming weights[0] is the input layer
        throw std::invalid_argument("Input size does not match the input layer size.");
    }
    return feedforward(input);
}

void MLP::save_model(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to save model.");
    }

    for (const auto& layer_weights : weights) {
        for (const auto& neuron_weights : layer_weights) {
            file.write(reinterpret_cast<const char*>(neuron_weights.data()), neuron_weights.size() * sizeof(float));
        }
    }

    for (const auto& layer_biases : biases) {
        file.write(reinterpret_cast<const char*>(layer_biases.data()), layer_biases.size() * sizeof(float));
    }

    file.close();
}

void MLP::load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to load model.");
    }

    for (auto& layer_weights : weights) {
        for (auto& neuron_weights : layer_weights) {
            file.read(reinterpret_cast<char*>(neuron_weights.data()), neuron_weights.size() * sizeof(float));
        }
    }

    for (auto& layer_biases : biases) {
        file.read(reinterpret_cast<char*>(layer_biases.data()), layer_biases.size() * sizeof(float));
    }

    file.close();
}

float MLP::activation_function(float x) {
    return 1.0f / (1.0f + std::exp(-x)); // Sigmoid function
}

float MLP::dot_product(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be of the same size for dot product.");
    }

    float result = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

std::vector<float> MLP::feedforward(const std::vector<float>& input) {
    std::vector<float> activations = input;

    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<float> new_activations(weights[i].size());

        for (size_t j = 0; j < weights[i].size(); ++j) {
            new_activations[j] = activation_function(dot_product(weights[i][j], activations) + biases[i][j]);
        }

        activations = new_activations;
    }

    return activations;
}
