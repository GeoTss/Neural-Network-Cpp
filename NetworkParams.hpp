#ifndef NETWORKPARAMS_HPP
#define NETWORKPARAMS_HPP
#pragma once

#include "NetworkDefs.hpp"
#include <vector>

template<typename T>
struct NetworkParams {
    double _learning_rate;
    size_t _batch_size;
    size_t _total_layers;
    size_t _epochs;
    LossFunctionType _loss_function_e;
    OptimizerType _optimizer_e;

    std::vector<size_t> _layers;
    std::vector<ActivationFunctionType> _activation_enums;
    std::vector<ActivationFunction<T>> _activation_functions;
    std::vector<ActivationFunctionDerivative<T>> _activation_functions_derivatives;

    std::vector<Matrix<T>> _weights;
    std::vector<Matrix<T>> _biases;
    std::vector<Matrix<T>> _activations;
    std::vector<Matrix<T>> _activation_derivatives;
    std::vector<Matrix<T>> _weight_derivatives;
    std::vector<Matrix<T>> _bias_derivatives;
};


#endif