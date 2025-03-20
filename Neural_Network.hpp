#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP
#pragma once

#include "NetworkParams.hpp"
#include "Optimizers.hpp"
#include "LossFunctions.hpp"
#include "ActivationFunctions.hpp"
#include "WeightInitializers.hpp"

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <map>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <concepts>

namespace NNUtils {

    static const std::map<ActivationFunctionType, WeightInitializerType> recommended_normal_activation_function_weight_initializer_map = {
        {ActivationFunctionType::Sigmoid, WeightInitializerType::XavierNormal},
        {ActivationFunctionType::ReLU, WeightInitializerType::HeNormal},
        {ActivationFunctionType::Tanh, WeightInitializerType::XavierNormal},
        {ActivationFunctionType::Softmax, WeightInitializerType::XavierNormal}
    };

    static const std::map<ActivationFunctionType, WeightInitializerType> recommended_uniform_activation_function_weight_initializer_map = {
        {ActivationFunctionType::Sigmoid, WeightInitializerType::XavierUniform},
        {ActivationFunctionType::ReLU, WeightInitializerType::HeUniform},
        {ActivationFunctionType::Tanh, WeightInitializerType::XavierUniform},
        {ActivationFunctionType::Softmax, WeightInitializerType::XavierUniform}
    };

};

template <typename T>
void checkFinite(const Matrix<T>& mat, const std::string& name) {
    if (!mat.array().isFinite().all()) {
        std::cerr << "Error: Matrix " << name << " contains inf or nan values!" << std::endl;
        assert(false && "Matrix contains inf or nan values.");
    }
}

template <typename T>
void checkDimensions(const Matrix<T>& mat, int rows, int cols, const std::string& name) {
    if (mat.rows() != rows || mat.cols() != cols) {
        std::cerr << "Error: Matrix " << name << " has incorrect dimensions. Expected: "
            << rows << "x" << cols << ", Got: " << mat.rows() << "x" << mat.cols() << std::endl;
        assert(false && "Matrix dimensions mismatch.");
    }
}

template <typename T = float, bool _Release = true>
    requires std::floating_point<T>
class NeuralNetwork {
private:
    const Matrix<T>& _forwardPropagation(const Matrix<T>& input) {
        _parameters._activations[0] = input;

        for (size_t i = 1; i < _parameters._layers.size(); ++i) {

            Matrix<T> z = _parameters._weights[i - 1] * _parameters._activations[i - 1] +
                _parameters._biases[i - 1] * Eigen::RowVectorX<T>::Ones(_parameters._activations[i - 1].cols());

            if constexpr (!_Release) {
                checkFinite(z, "Z (Layer " + std::to_string(i) + ")");
                checkDimensions(z, _parameters._layers[i], _parameters._activations[i - 1].cols(), "Z (Layer " + std::to_string(i) + ")");
            }

            _parameters._activations[i].noalias() = _parameters._activation_functions[i - 1](z);
            if constexpr (!_Release) {
                checkFinite(_parameters._activations[i], "Activations (Layer " + std::to_string(i) + ")");
            }

            _parameters._activation_derivatives[i - 1].noalias() = _parameters._activation_functions_derivatives[i - 1](z);
            if constexpr (!_Release) {
                checkFinite(_parameters._activation_derivatives[i - 1], "Activations (Layer " + std::to_string(i) + ")");
            }
        }

        return _parameters._activations.back();
    }

    void _backPropagation(const Matrix<T>& output, const Matrix<T>& input, const Matrix<T>& target) {

        int m = input.cols();

        if constexpr (!_Release) {
            checkFinite(output, "Output");
            checkFinite(target, "Target");
            checkDimensions(output, _parameters._layers.back(), m, "Output");
            checkDimensions(target, _parameters._layers.back(), m, "Target");
        }

        Matrix<T> dz = _loss_function_derivative(output, target);
        if constexpr (!_Release) {
            checkFinite(dz, "DZ (Output Layer)");
        }

        _parameters._weight_derivatives.back().noalias() = (dz * _parameters._activations[_parameters._layers.size() - 2].transpose()) / m;
        _parameters._bias_derivatives.back().noalias() = dz.rowwise().sum() / m;

        for (int i = _parameters._layers.size() - 2; i > 0; --i) {
            dz.noalias() = (_parameters._weights[i].transpose() * dz).cwiseProduct(_parameters._activation_derivatives[i - 1]);
            if constexpr (!_Release) {
                checkFinite(dz, "DZ (Layer " + std::to_string(i) + ")");
            }

            _parameters._weight_derivatives[i - 1].noalias() = (dz * _parameters._activations[i - 1].transpose()) / m;
            _parameters._bias_derivatives[i - 1].noalias() = dz.rowwise().sum() / m;
        }
    }

public:

    NetworkParams<T> _parameters;
    LossFunction<T> _loss_function;
    LossFunctionDerivative<T> _loss_function_derivative;
    std::shared_ptr<BaseOptimizer<T>> _optimizer;

    NeuralNetwork() = default;
    NeuralNetwork(std::vector<size_t> layers, std::vector<ActivationFunctionType> activation_functions, LossFunctionType loss_function, OptimizerType optimizer, std::vector<WeightInitializerType> initializers, double learning_rate, size_t batch_size) {
        _parameters._total_layers = layers.size();
        std::cout << "Total layers: " << _parameters._total_layers << std::endl;
        _parameters._learning_rate = learning_rate;
        _parameters._batch_size = batch_size;

        _parameters._loss_function_e = loss_function;
        _parameters._optimizer_e = optimizer;

        _parameters._activation_enums = activation_functions;

        _parameters._layers = layers;
        _parameters._activations.resize(_parameters._total_layers);
        _parameters._activation_derivatives.resize(_parameters._total_layers);
        _parameters._weights.resize(_parameters._total_layers - 1);
        _parameters._biases.resize(_parameters._total_layers - 1);
        _parameters._weight_derivatives.resize(_parameters._total_layers - 1);
        _parameters._bias_derivatives.resize(_parameters._total_layers - 1);

        for (const auto& activation_function : activation_functions) {
            const auto [activation_function_type, activation_function_derivative_type] = NNUtils::activation_functions_map<T>.at(activation_function);
            _parameters._activation_functions.push_back(activation_function_type);
            _parameters._activation_functions_derivatives.push_back(activation_function_derivative_type);
        }

        for (size_t i = 0; i < _parameters._layers.size() - 1; ++i) {
            WeightInitializer<T> initializer;
            switch (initializers[i]) {
            case WeightInitializerType::RecommendedNormal:
                initializer = NNUtils::weight_initializers_map<T>.at(NNUtils::recommended_normal_activation_function_weight_initializer_map.at(activation_functions[i]));
                break;
            case WeightInitializerType::RecommendedUniform:
                initializer = NNUtils::weight_initializers_map<T>.at(NNUtils::recommended_uniform_activation_function_weight_initializer_map.at(activation_functions[i]));
                break;
            default:
                initializer = NNUtils::weight_initializers_map<T>.at(initializers[i]);
                break;
            }

            _parameters._weights[i] = initializer(_parameters._layers[i + 1], _parameters._layers[i]);
            _parameters._biases[i] = Matrix<T>::Zero(_parameters._layers[i + 1], 1);
            _parameters._weight_derivatives[i] = Matrix<T>::Zero(_parameters._layers[i + 1], _parameters._layers[i]);
            _parameters._bias_derivatives[i] = Matrix<T>::Zero(_parameters._layers[i + 1], 1);

            assert(_parameters._weights[i].rows() == _parameters._layers[i + 1]);
            assert(_parameters._weights[i].cols() == _parameters._layers[i]);
            assert(_parameters._biases[i].rows() == _parameters._layers[i + 1]);
            assert(_parameters._biases[i].cols() == 1);

            double max_weight = _parameters._weights[i].array().abs().maxCoeff();
            assert(max_weight < 10.0);
            std::cout << "Layer " << i + 1 << ": Weights initialized with max value " << max_weight << std::endl;
        }

        const auto& loss_functions = NNUtils::loss_functions_map<T>.at(loss_function);
        _loss_function = loss_functions.first;
        _loss_function_derivative = loss_functions.second;

        switch (optimizer) {
        case OptimizerType::SGD:
            _optimizer = std::make_unique<SGD_Optimizer<T>>();
            break;
        case OptimizerType::Adam:
            _optimizer = std::make_unique<Adam_Optimizer<T>>();
            break;
        default:
            throw std::invalid_argument("Unsupported optimizer type");
        }
        _optimizer->init(_parameters);
    }

    int getOneHotMatches(const Matrix<T>& output, const Matrix<T>& target) {
        int correct_predictions{};
        Eigen::Index maxRow, maxCol;
        for (int col = 0; col < output.cols(); ++col) {
            output.col(col).maxCoeff(&maxRow, &maxCol);
            if (target(maxRow, col) == 1.0) {
                ++correct_predictions;
            }
        }

        return correct_predictions;
    }

    void train(const Matrix<T>& input, const Matrix<T>& target, size_t epochs) {

        if constexpr (!_Release) {
            checkFinite(input, "Training Input");
            checkFinite(target, "Training Target");
            checkDimensions(input, _parameters._layers[0], input.cols(), "Training Input");
            checkDimensions(target, _parameters._layers.back(), target.cols(), "Training Target");
        }

        _parameters._epochs = epochs;

        int num_batches = input.cols() / _parameters._batch_size;

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double total_cost = 0.0;
            size_t correct_predictions = 0;

            for (size_t i = 0; i < input.cols(); i += _parameters._batch_size) {
                Matrix<T> input_batch = input.middleCols(i, std::min(_parameters._batch_size, input.cols() - i));
                Matrix<T> target_batch = target.middleCols(i, std::min(_parameters._batch_size, target.cols() - i));

                const Matrix<T>& output = _forwardPropagation(input_batch);
                _backPropagation(output, input_batch, target_batch);
                _optimizer->update(_parameters);

                total_cost += _loss_function(output, target_batch);

                correct_predictions += getOneHotMatches(output, target_batch);
            }
            total_cost /= num_batches;
            double accuracy = static_cast<double>(correct_predictions) / input.cols();
            std::cout << "Epoch: " << epoch << " Cost: " << total_cost << " Accuracy: " << accuracy * 100.0 << "%" << std::endl;
        }
    }

    int test(const Matrix<T>& test_input, const Matrix<T>& test_labels) {
        const Matrix<T>& output = _forwardPropagation(test_input);
        int correctCount = getOneHotMatches(output, test_labels);
        return correctCount;
    }

    void saveModel(const std::filesystem::path filename) {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return;
        }
        outFile.clear();
        outFile.write(reinterpret_cast<const char*>(&_parameters._total_layers), sizeof(size_t));
        outFile.write(reinterpret_cast<const char*>(&_parameters._learning_rate), sizeof(double));
        outFile.write(reinterpret_cast<const char*>(&_parameters._batch_size), sizeof(size_t));
        outFile.write(reinterpret_cast<const char*>(&_parameters._epochs), sizeof(size_t));

        auto loss_e = static_cast<std::underlying_type_t<LossFunctionType>>(_parameters._loss_function_e);
        outFile.write(reinterpret_cast<const char*>(&loss_e), sizeof(std::underlying_type_t<LossFunctionType>));

        auto optimizer_e = static_cast<std::underlying_type_t<OptimizerType>>(_parameters._optimizer_e);
        outFile.write(reinterpret_cast<const char*>(&optimizer_e), sizeof(std::underlying_type_t<OptimizerType>));

        outFile.write(reinterpret_cast<const char*>(_parameters._layers.data()), sizeof(size_t) * _parameters._total_layers);

        for (size_t i = 0; i < _parameters._activation_enums.size(); ++i) {
            auto activation_enum_value = static_cast<std::underlying_type_t<ActivationFunctionType>>(_parameters._activation_enums[i]);
            outFile.write(reinterpret_cast<const char*>(&activation_enum_value), sizeof(std::underlying_type_t<ActivationFunctionType>));
        }


        for (size_t i = 0; i < _parameters._total_layers - 1; ++i) {
            Eigen::Index rows = _parameters._weights[i].rows();
            Eigen::Index cols = _parameters._weights[i].cols();
            outFile.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
            outFile.write(reinterpret_cast<const char*>(&cols), sizeof(Eigen::Index));
            outFile.write(reinterpret_cast<const char*>(_parameters._weights[i].data()), sizeof(T) * rows * cols);

            rows = _parameters._biases[i].rows();
            cols = _parameters._biases[i].cols();
            outFile.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
            outFile.write(reinterpret_cast<const char*>(&cols), sizeof(Eigen::Index));
            outFile.write(reinterpret_cast<const char*>(_parameters._biases[i].data()), sizeof(T) * rows * cols);
        }

        outFile.close();
    }

    void loadModel(const std::filesystem::path filename) {
        std::ifstream inFile(filename, std::ios::binary);
        if (!inFile) {
            std::cerr << "Error opening file for reading: " << filename << std::endl;
            return;
        }

        try {
            std::cout << "Reading model data..." << std::endl;

            // Read total layers
            inFile.read(reinterpret_cast<char*>(&_parameters._total_layers), sizeof(size_t));
            if (inFile.eof() || _parameters._total_layers <= 0) {
                std::cerr << "Invalid or corrupted total layers value." << std::endl;
                return;
            }
            _parameters._activations.resize(_parameters._total_layers);
            _parameters._activation_derivatives.resize(_parameters._total_layers);
            _parameters._weights.resize(_parameters._total_layers - 1);
            _parameters._biases.resize(_parameters._total_layers - 1);
            _parameters._weight_derivatives.resize(_parameters._total_layers - 1);
            _parameters._bias_derivatives.resize(_parameters._total_layers - 1);
            // Read basic parameters
            // inFile.read(reinterpret_cast<char*>(&_parameters._total_layers), sizeof(size_t));
            inFile.read(reinterpret_cast<char*>(&_parameters._learning_rate), sizeof(double));
            inFile.read(reinterpret_cast<char*>(&_parameters._batch_size), sizeof(size_t));
            inFile.read(reinterpret_cast<char*>(&_parameters._epochs), sizeof(size_t));

            std::underlying_type_t<LossFunctionType> loss_e;
            inFile.read(reinterpret_cast<char*>(&loss_e), sizeof(std::underlying_type_t<LossFunctionType>));
            
            _parameters._loss_function_e = static_cast<LossFunctionType>(loss_e);
            const auto& loss_functions = NNUtils::loss_functions_map<T>.at(_parameters._loss_function_e);
            _loss_function = loss_functions.first;
            _loss_function_derivative = loss_functions.second;

            std::underlying_type_t<OptimizerType> optimizer_e;
            inFile.read(reinterpret_cast<char*>(&optimizer_e), sizeof(std::underlying_type_t<OptimizerType>));
            _parameters._optimizer_e = static_cast<OptimizerType>(optimizer_e);
            _optimizer = NNUtils::optimizers_map<T>.at(_parameters._optimizer_e);

            // Read layer sizes
            _parameters._layers.resize(_parameters._total_layers);
            inFile.read(reinterpret_cast<char*>(_parameters._layers.data()), sizeof(size_t) * _parameters._total_layers);
            for (size_t i = 0; i < _parameters._total_layers; ++i) {
                if (_parameters._layers[i] <= 0 || _parameters._layers[i] > 1e6) {
                    std::cerr << "Invalid layer size for layer " << i << ": " << _parameters._layers[i] << std::endl;
                    return;
                }
            }

            // Read activation enums
            _parameters._activation_enums.resize(_parameters._total_layers - 1);
            for (size_t i = 0; i < _parameters._total_layers - 1; ++i) {
                std::underlying_type_t<ActivationFunctionType> activation_enum_value;
                inFile.read(reinterpret_cast<char*>(&activation_enum_value), sizeof(std::underlying_type_t<ActivationFunctionType>));
                _parameters._activation_enums[i] = static_cast<ActivationFunctionType>(activation_enum_value);
            }
            std::cout << "Loaded activation functions.\n";

            // Reinitialize activation functions
            _parameters._activation_functions.clear();
            _parameters._activation_functions_derivatives.clear();
            for (const auto& activation_function : _parameters._activation_enums) {
                const auto [activation_function_type, activation_function_derivative_type] =
                    NNUtils::activation_functions_map<T>.at(activation_function);

                _parameters._activation_functions.push_back(activation_function_type);
                _parameters._activation_functions_derivatives.push_back(activation_function_derivative_type);
            }
            std::cout << "Initialized activation functions.\n";

            // Read weights and biases
            _parameters._weights.resize(_parameters._total_layers - 1);
            _parameters._biases.resize(_parameters._total_layers - 1);
            for (size_t i = 0; i < _parameters._total_layers - 1; ++i) {
                Eigen::Index rows, cols;

                // Read weights
                inFile.read(reinterpret_cast<char*>(&rows), sizeof(Eigen::Index));
                inFile.read(reinterpret_cast<char*>(&cols), sizeof(Eigen::Index));
                if (rows <= 0 || cols <= 0 || rows > 1e6 || cols > 1e6) {
                    std::cerr << "Invalid weights dimensions for layer " << i << ": " << rows << "x" << cols << std::endl;
                    return;
                }
                _parameters._weights[i].resize(rows, cols);
                inFile.read(reinterpret_cast<char*>(_parameters._weights[i].data()), sizeof(T) * rows * cols);

                // Read biases
                inFile.read(reinterpret_cast<char*>(&rows), sizeof(Eigen::Index));
                inFile.read(reinterpret_cast<char*>(&cols), sizeof(Eigen::Index));
                if (rows <= 0 || cols != 1) {
                    std::cerr << "Invalid biases dimensions for layer " << i << ": " << rows << "x" << cols << std::endl;
                    return;
                }
                _parameters._biases[i].resize(rows, cols);
                inFile.read(reinterpret_cast<char*>(_parameters._biases[i].data()), sizeof(T) * rows * cols);
            
                _parameters._weight_derivatives[i] = Matrix<T>::Zero(_parameters._layers[i + 1], _parameters._layers[i]);
                _parameters._bias_derivatives[i] = Matrix<T>::Zero(_parameters._layers[i + 1], 1);
            }

            inFile.close();
            std::cout << "Model loaded successfully." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception occurred while loading model: " << e.what() << std::endl;
        }
    }

    void printNetworkInfo() const {
        std::cout << "Neural Network Information:\n";
        std::cout << "Total Layers: " << _parameters._total_layers << "\n";
        std::cout << "Learning Rate: " << _parameters._learning_rate << "\n";
        std::cout << "Batch Size: " << _parameters._batch_size << "\n";
        std::cout << "Epochs: " << _parameters._epochs << "\n";
        std::cout << "Loss Function: " << static_cast<int>(_parameters._loss_function_e) << "\n";
        std::cout << "Optimizer: " << static_cast<int>(_parameters._optimizer_e) << "\n";

        std::cout << "Layer Sizes:\n";
        for (size_t i = 0; i < _parameters._layers.size(); ++i) {
            std::cout << "  Layer " << i << ": " << _parameters._layers[i] << " neurons\n";
        }

        std::cout << "Activation Functions:\n";
        for (size_t i = 0; i < _parameters._activation_enums.size(); ++i) {
            std::cout << "  Layer " << i + 1 << ": " << static_cast<int>(_parameters._activation_enums[i]) << "\n";
        }

        std::cout << "Weights and Biases Dimensions:\n";
        for (size_t i = 0; i < _parameters._weights.size(); ++i) {
            std::cout << "  Layer " << i + 1 << " Weights: " 
                      << _parameters._weights[i].rows() << "x" << _parameters._weights[i].cols() << "\n";
            std::cout << "  Layer " << i + 1 << " Biases: " 
                      << _parameters._biases[i].rows() << "x" << _parameters._biases[i].cols() << "\n";
        }
    }
};

#endif