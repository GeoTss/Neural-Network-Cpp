#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP
#pragma once

#include <concepts>
#include <map>
#include <memory>
#include "NetworkParams.hpp"
#include "NetworkDefs.hpp"

template<typename T>
    requires std::floating_point<T>
struct BaseOptimizer {
    virtual void init(const NetworkParams<T>&) {}
    virtual void update(NetworkParams<T>&) = 0;

    virtual ~BaseOptimizer() = default;
};

template<typename T>
struct SGD_Optimizer : public BaseOptimizer<T> {

    void update(NetworkParams<T>& nn) override {
        for (size_t i = 0; i < nn._weights.size(); ++i) {
            nn._weights[i].noalias() -= nn._learning_rate * nn._weight_derivatives[i];
            nn._biases[i].noalias() -= nn._learning_rate * nn._bias_derivatives[i];
        }
    };
};

template<typename T>
struct Adam_Optimizer : public BaseOptimizer<T> {

    std::vector<Matrix<T>> _m_weights;
    std::vector<Matrix<T>> _v_weights;
    std::vector<Matrix<T>> _m_biases;
    std::vector<Matrix<T>> _v_biases;
    static constexpr double _beta1 = 0.9;
    static constexpr double _beta2 = 0.999;
    static constexpr double _epsilon = 1e-8;
    size_t _t = 0;

    void init(const NetworkParams<T>& nn) override {
        _m_weights.clear();
        _v_weights.clear();
        _m_biases.clear();
        _v_biases.clear();

        for (size_t i = 0; i < nn._weights.size(); ++i) {
            _m_weights.push_back(Matrix<T>::Zero(nn._weights[i].rows(), nn._weights[i].cols()));
            _v_weights.push_back(Matrix<T>::Zero(nn._weights[i].rows(), nn._weights[i].cols()));
            _m_biases.push_back(Matrix<T>::Zero(nn._weights[i].rows(), 1));
            _v_biases.push_back(Matrix<T>::Zero(nn._weights[i].rows(), 1));
        }
    }

    void update(NetworkParams<T>& nn) override {
        _t++;
        for (size_t i = 0; i < nn._weights.size(); ++i) {
            _m_weights[i].noalias() = _beta1 * _m_weights[i] + (1 - _beta1) * nn._weight_derivatives[i];
            _v_weights[i].noalias() = _beta2 * _v_weights[i] + (1 - _beta2) * nn._weight_derivatives[i].cwiseProduct(nn._weight_derivatives[i]);

            Matrix<T> m_weight_hat = _m_weights[i] / (1 - std::pow(_beta1, _t));
            Matrix<T> v_weight_hat = _v_weights[i] / (1 - std::pow(_beta2, _t));

            _m_biases[i].noalias() = _beta1 * _m_biases[i] + (1 - _beta1) * nn._bias_derivatives[i];
            _v_biases[i].noalias() = _beta2 * _v_biases[i] + (1 - _beta2) * nn._bias_derivatives[i].cwiseProduct(nn._bias_derivatives[i]);

            Matrix<T> m_bias_hat = _m_biases[i] / (1 - std::pow(_beta1, _t));
            Matrix<T> v_bias_hat = _v_biases[i] / (1 - std::pow(_beta2, _t));

            nn._weights[i].noalias() -= (nn._learning_rate * m_weight_hat.array() / (v_weight_hat.array().sqrt() + _epsilon)).matrix();
            nn._biases[i].noalias() -= (nn._learning_rate * m_bias_hat.array() / (v_bias_hat.array().sqrt() + _epsilon)).matrix();
        }
    }
};

namespace NNUtils{

    template<typename T>
    static std::map<OptimizerType, std::shared_ptr<BaseOptimizer<T>>> optimizers_map = {
        {
            OptimizerType::SGD,
            std::make_shared<SGD_Optimizer<T>>()
        },
        {
            OptimizerType::Adam,
            std::make_shared<Adam_Optimizer<T>>()
        }
    };
}

#endif