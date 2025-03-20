#ifndef LOSS_FUNCTIONS_HPP
#define LOSS_FUNCTIONS_HPP
#pragma once

#include "NetworkDefs.hpp"
#include <map>

namespace NNUtils{

    static const double epsilon = 1e-8;
    
    template<typename T>
    static const std::map<LossFunctionType, std::pair<LossFunction<T>, LossFunctionDerivative<T>>> loss_functions_map = {
        {
            LossFunctionType::MeanSquaredError,
            {
                [](const Matrix<T>& y, const Matrix<T>& t) -> double {
                    return (y - t).array().square().sum() / t.cols();
                },
                [](const Matrix<T>& y, const Matrix<T>& t) -> Matrix<T> {
                    return 2 * (y - t) / t.cols();
                }
            }
        },
        {
            LossFunctionType::CrossEntropy,
            {
                [](const Matrix<T>& output, const Matrix<T>& target) -> double {
                    Matrix<T> safe_output = output.array().cwiseMax(epsilon);
                    return -(target.array() * safe_output.array().log()).sum() / target.cols();
                },
                [](const Matrix<T>& output, const Matrix<T>& target) -> Matrix<T> {
                    return output - target;
                }
            }
        }
    };

};



#endif