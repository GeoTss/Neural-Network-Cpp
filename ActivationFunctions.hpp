#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP
#pragma once

#include "NetworkDefs.hpp"
#include <map>

namespace NNUtils{
    template<typename T>
    static const std::map<ActivationFunctionType, std::pair<ActivationFunction<T>, ActivationFunctionDerivative<T>>> activation_functions_map = {
        {
            ActivationFunctionType::Sigmoid,
            {
                [](const Matrix<T>& x) {return 1.0 / (1.0 + (-x.array()).exp());},
                [](const Matrix<T>& x) {return x.array() * (1.0 - x.array());}
            }
        },
        {
            ActivationFunctionType::ReLU,
            {
                [](const Matrix<T>& x) {return x.cwiseMax(0.);},
                [](const Matrix<T>& x) {return (x.array() > 0).template cast<T>();}
            }
        },
        {
            ActivationFunctionType::Tanh,
            {
                [](const Matrix<T>& x) {return x.array().tanh();},
                [](const Matrix<T>& x) {return 1.0 - x.array().tanh().square();}
            }
        },
        {
            ActivationFunctionType::Softmax,
            {
                [](const Matrix<T>& Z) {
                    Matrix<T> Z_stable = Z.rowwise() - Z.colwise().maxCoeff();
                    Matrix<T> expZ = Z_stable.array().exp();
                    Eigen::RowVectorX<T> sumExp = expZ.colwise().sum();

                    for (int j = 0; j < expZ.cols(); ++j)
                        expZ.col(j) /= sumExp(j);
                    return expZ;
                },
                [](const Matrix<T>& Z) { return Z; }
            }
        }
    };

};

#endif