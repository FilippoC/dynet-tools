#pragma once

#include <memory>
#include "dynet/dict.h"
#include "dynet/cfsm-builder.h"
#include "dytools/builders/builder.h"

namespace dytools
{

struct TaggerSettings
{
    bool output_bias = false;
    unsigned dim = 128u;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & output_bias;
        ar & dim;
    }
};

struct TaggerBuilder : public Builder
{
    const TaggerSettings settings;
    dynet::ParameterCollection local_pc;
    std::shared_ptr<dynet::Dict> dict;

    dynet::StandardSoftmaxBuilder builder;

    TaggerBuilder(dynet::ParameterCollection& pc, const TaggerSettings& settings, std::shared_ptr<dynet::Dict> dict, unsigned dim_input);
    void new_graph(dynet::ComputationGraph& cg, bool update = true);

    dynet::Expression operator()(const dynet::Expression& input);
    dynet::Expression operator()(const std::vector<dynet::Expression>& input);
};

}