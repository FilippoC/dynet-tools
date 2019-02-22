#pragma once

#include <vector>

#include "dynet/model.h"
#include "dynet/expr.h"
#include "dytools/activation.h"

namespace dytools
{

struct MLPSettings
{
    unsigned dim = 128;
    unsigned layers = 1;
    ActivationType activation = ActivationType::relu;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        ar & dim;
        ar & layers;
    }
};

struct MLPBuilder
{
    const MLPSettings settings;
    dynet::ParameterCollection local_pc;

    std::vector<dynet::Parameter> p_W, p_bias;
    std::vector<dynet::Expression> e_W, e_bias;

    unsigned _output_rows;
    bool _training = true;
    float dropout_rate = 0.f;

    MLPBuilder(dynet::ParameterCollection& pc, const MLPSettings& settings, unsigned dim_input);

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);
    dynet::Expression apply(const dynet::Expression &input);

    void set_dropout(float value);
    unsigned output_rows() const;
};


}