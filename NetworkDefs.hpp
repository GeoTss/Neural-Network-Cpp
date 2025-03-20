#ifndef NETWORK_DEFS_HPP
#define NETWORK_DEFS_HPP
#pragma once

#include <Eigen/Core>
#include <functional>

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
using ActivationFunction = std::function<Matrix<T>(const Matrix<T>&)>;
template<typename T>
using ActivationFunctionDerivative = std::function<Matrix<T>(const Matrix<T>&)>;

template<typename T>
using LossFunction = std::function<double(const Matrix<T>&, const Matrix<T>&)>;

template<typename T>
using LossFunctionDerivative = std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)>;

template<typename T>
using WeightInitializer = std::function<Matrix<T>(size_t, size_t)>;

enum struct ActivationFunctionType { Sigmoid, ReLU, Tanh, Softmax };
enum struct OptimizerType { SGD, Adam, Custom };
enum struct LossFunctionType { MeanSquaredError, CrossEntropy };
enum struct WeightInitializerType { Zero, RandomNormal, XavierNormal, HeNormal, RandomUniform, XavierUniform, HeUniform, RecommendedNormal, RecommendedUniform };


#endif