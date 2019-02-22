#pragma once

#include "dynet/expr.h"

namespace dytools
{

enum struct ActivationType
{
    relu,
    tanh
};

inline dynet::Expression activation(const dynet::Expression& e, const ActivationType type)
{
    if (type == ActivationType::relu)
        return dynet::rectify(e);
    else
        return dynet::tanh(e);
}

}