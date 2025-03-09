#pragma once

#include <string>
#include <vector>

class MLP {
public:
  MLP(const std::vector<int> &layers, float learning_rate);
  void train(const std::vector<std::vector<float>> &inputs,
             const std::vector<std::vector<float>> &labels, int epochs);
  std::vector<float> predict(const std::vector<float> &input);
  void save_model(const std::string &filename);
  void load_model(const std::string &filename);

private:
  inline float activation_function(float x);
  inline float activation_derivative(float x);
  float dot_product(const std::vector<float> &v1, const std::vector<float> &v2);
  std::vector<float> feedforward(const std::vector<float> &input);

  std::vector<std::vector<std::vector<float>>> weights;
  std::vector<std::vector<float>> biases;
  float learning_rate;
  float average_loss = 0.0f; // To track the average loss
};
