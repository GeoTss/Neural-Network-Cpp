#ifndef WEIGHT_INITIALIZERS_HPP
#define WEIGHT_INITIALIZERS_HPP
#pragma once

#include "NetworkDefs.hpp"
#include <map>
#include <random>

namespace NNUtils{

    std::random_device device;
    std::mt19937 generator(device());

    template<typename T>
    static const std::map<WeightInitializerType, WeightInitializer<T>> weight_initializers_map = {
        {
            WeightInitializerType::Zero,
            [](size_t out, size_t in) {return Matrix<T>::Zero(out, in);}
        },
        {
            WeightInitializerType::RandomUniform,
            [](size_t out, size_t in) {
                std::uniform_real_distribution dist(0., 1.);
                Matrix<T> W(out, in);
                for (int i = 0; i < out; ++i)
                    for (int j = 0; j < in; ++j)
                        W(i, j) = dist(generator);
                return W;
            }
        },
        {
            WeightInitializerType::XavierUniform,
            [](size_t out, size_t in) {
                std::uniform_real_distribution dist(-std::sqrt(6.0 / (out + in)), std::sqrt(6.0 / (out + in)));
                Matrix<T> W(out, in);
                for (int i = 0; i < out; ++i)
                    for (int j = 0; j < in; ++j)
                        W(i, j) = dist(generator);
                return W;
            }
        },
        {
            WeightInitializerType::HeUniform,
            [](size_t out, size_t in) {
                std::uniform_real_distribution dist(-std::sqrt(2.0 / in), std::sqrt(2.0 / out));
                Matrix<T> W(out, in);
                for (int i = 0; i < out; ++i)
                    for (int j = 0; j < in; ++j)
                        W(i, j) = dist(generator);
                return W;
            }
        },
        {
            WeightInitializerType::RandomNormal,
            [](size_t out, size_t in) {
                std::normal_distribution dist(0., 1.);
                Matrix<T> W(out, in);
                for (int i = 0; i < out; ++i)
                    for (int j = 0; j < in; ++j)
                        W(i, j) = dist(generator);
                return W;
            }
        },
        {
            WeightInitializerType::XavierNormal,
            [](size_t out, size_t in) {
                std::normal_distribution dist(0., std::sqrt(6.0 / (out + in)));
                Matrix<T> W(out, in);
                for (int i = 0; i < out; ++i)
                    for (int j = 0; j < in; ++j)
                        W(i, j) = dist(generator);
                return W;
            }
        },
        {
            WeightInitializerType::HeNormal,
            [](size_t out, size_t in) {
                std::normal_distribution dist(0., std::sqrt(2.0 / in));
                Matrix<T> W(out, in);
                for (int i = 0; i < out; ++i)
                    for (int j = 0; j < in; ++j)
                        W(i, j) = dist(generator);
                return W;
            }
        }
    };

};

#endif