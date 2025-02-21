#include "Mlp.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

glm::vec4 MLP::activate_derivative(const glm::vec4 &x) {
  return glm::vec4(x.x > 0 ? 1.0f : 0.0f, x.y > 0 ? 1.0f : 0.0f,
                   x.z > 0 ? 1.0f : 0.0f, x.w > 0 ? 1.0f : 0.0f);
}

MLP::MLP(int input_size, int output_size, std::vector<int> hidden_layers,
         float lr)
    : learning_rate(lr), input_size(input_size), output_size(output_size),
      hidden_layers(hidden_layers) {

  previous_size = input_size; // Initialize previous_size with the input size

  // Initialize weights and biases for hidden layers
  for (int neurons : hidden_layers) {
    glm::mat4 weight_matrix =
        glm::mat4(0.0f); // Create weight matrix for the layer
    // Randomly initialize weights here as needed
    weights.push_back(weight_matrix);
    biases.push_back(glm::vec4(0.0f)); // Initialize biases to zero
    previous_size = neurons; // Update previous_size for the next layer
  }

  // Initialize weights and biases for the output layer
  glm::mat4 output_weight_matrix =
      glm::mat4(0.0f); // Create weight matrix for the output layer
  // Randomly initialize weights here as needed
  weights.push_back(output_weight_matrix);
  biases.push_back(glm::vec4(0.0f)); // Initialize biases for output layer
}

// ReLU activation function
glm::vec4 MLP::activate(const glm::vec4 &x) {
  return glm::vec4(std::max(0.0f, x.x), std::max(0.0f, x.y),
                   std::max(0.0f, x.z), std::max(0.0f, x.w));
}

// Feedforward function
glm::vec4 MLP::feedforward(const glm::vec4 &input) {
  glm::vec4 output = input;

  for (size_t layer = 0; layer < weights.size(); ++layer) {
    output = feedforward_internal(
        output); // Pass the output of the previous layer to the next
  }

  return output; // Final output after all layers
}

glm::vec4 MLP::feedforward_internal(const glm::vec4 &input) {
  glm::vec4 weighted_sum = input; // Start with the input for the first layer

  for (size_t layer = 0; layer < weights.size(); ++layer) {
    weighted_sum = weights[layer] * weighted_sum + biases[layer];
    weighted_sum = activate(weighted_sum); // Apply activation after each layer
  }

  return weighted_sum;
}

// Training method
void MLP::train(const std::vector<glm::vec4> &inputs,
                const std::vector<glm::vec4> &targets, int epochs) {
  {
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
      float total_loss = 0.0f;

      for (size_t i = 0; i < inputs.size(); ++i) {
        glm::vec4 input = inputs[i];
        glm::vec4 target = targets[i];

        // Forward pass
        glm::vec4 output = feedforward(input);

        // Calculate loss (MSE)
        float loss = glm::dot(output - target, output - target);
        total_loss += loss;

        // Backpropagation
        glm::vec4 error = output - target;    // Calculate output error
        total_loss += glm::dot(error, error); // Add to total loss

        // Backpropagation loop
        for (size_t layer = weights.size(); layer > 0; --layer) {
          // Apply derivative of activation function for delta
          glm::vec4 delta =
              error *
              activate_derivative(output); // Adjust for the correct output

          // Update weights and biases
          if (layer > 1) {
            glm::vec4 input_to_layer =
                inputs[i]; // Save the input for weight updates
            weights[layer - 1] -=
                learning_rate * glm::outerProduct(delta, input_to_layer);
            biases[layer - 1] -= learning_rate * delta;

            // Calculate the error for the next layer
            error = glm::transpose(weights[layer - 1]) * delta;
          } else {
            weights[layer - 1] -=
                learning_rate * glm::outerProduct(delta, input);
            biases[layer - 1] -= learning_rate * delta;
          }
        }

        // Print the average loss for the epoch
        std::cout << "Epoch " << epoch
                  << ", Average Loss: " << total_loss / inputs.size()
                  << std::endl;
      }
    }
  }
}

// Load weights from a file
void MLP::load_weights(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Unable to open file for loading weights.");
  }
  for (auto &weight_matrix : weights) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        file >> weight_matrix[i][j];
      }
    }
  }

  for (auto &bias_vector : biases) {
    for (int i = 0; i < 4; ++i) {
      file >> bias_vector[i];
    }
  }
}

// Save weights to a file
void MLP::save_weights(const std::string &filename) {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Unable to open file for saving weights.");
  }

  // Save weights
  for (size_t layer = 0; layer < weights.size(); ++layer) {
    for (int i = 0; i < weights[layer].length();
         ++i) { // Adjust for actual size
      for (int j = 0; j < weights[layer].length(); ++j) {
        file << weights[layer][i][j] << " ";
      }
      file << std::endl;
    }
  }

  // Save biases
  for (size_t layer = 0; layer < biases.size(); ++layer) {
    for (int i = 0; i < biases[layer].length(); ++i) { // Adjust for actual size
      file << biases[layer][i] << " ";
    }
    file << std::endl;
  }
}
